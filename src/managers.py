# src/managers.py
import numpy as np
import mujoco
from src.utils import get_body_geoms, save_simulation_state, restore_simulation_state

# --- Collision Calculator (Upgraded to OBB) ---
class CollisionCalculator:
    def __init__(self):
        # 移除快取，確保縮放即時生效
        pass
    
    def get_local_bounds(self, model, geom_id):
        """計算 Local AABB (未旋轉前的長寬高)"""
        g_type = model.geom_type[geom_id]
        
        # 1. 針對 Mesh (從頂點計算)
        if g_type == mujoco.mjtGeom.mjGEOM_MESH:
            dataid = model.geom_dataid[geom_id]
            if dataid != -1:
                vert_adr = model.mesh_vertadr[dataid]
                vert_num = model.mesh_vertnum[dataid]
                verts = model.mesh_vert[vert_adr : vert_adr + vert_num]
                
                if len(verts) > 0:
                    min_v = np.min(verts, axis=0)
                    max_v = np.max(verts, axis=0)
                    return min_v, max_v
                    
        # 2. 針對基本幾何體
        size = model.geom_size[geom_id]
        if g_type == mujoco.mjtGeom.mjGEOM_BOX:
            return -size[:3], size[:3]
        elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = size[0]
            return np.array([-r]*3), np.array([r]*3)
        else:
            r = model.geom_rbound[geom_id]
            return np.array([-r]*3), np.array([r]*3)

    def get_obb(self, model, data, geom_id):
        """
        取得 OBB (導向包圍盒) 的參數：
        Center: 世界座標中心點
        Axes: 3個旋轉軸向量 (3x3 矩陣)
        Extents: 半長/半寬/半高 (3,)
        """
        min_local, max_local = self.get_local_bounds(model, geom_id)
        
        # 1. 計算 Local 中心點與半徑
        center_local = (min_local + max_local) * 0.5
        extents = (max_local - min_local) * 0.5
        
        # 2. 讀取 MuJoCo 的位置與旋轉矩陣
        pos = data.geom_xpos[geom_id] # 世界座標原點
        mat = data.geom_xmat[geom_id].reshape(3, 3) # 旋轉矩陣 (這就是 OBB 的軸)
        
        # 3. 計算 OBB 在世界座標的真實中心 (考慮 Local offset)
        center_world = pos + mat @ center_local
        
        return center_world, mat, extents

    # 為了兼容舊代碼，保留 get_world_aabb 但不使用
    def get_world_aabb(self, model, data, geom_id):
        min_v, max_v = self.get_local_bounds(model, geom_id)
        corners = np.array([
            [min_v[0], min_v[1], min_v[2]], [min_v[0], min_v[1], max_v[2]],
            [min_v[0], max_v[1], min_v[2]], [min_v[0], max_v[1], max_v[2]],
            [max_v[0], min_v[1], min_v[2]], [max_v[0], min_v[1], max_v[2]],
            [max_v[0], max_v[1], min_v[2]], [max_v[0], max_v[1], max_v[2]]
        ])
        pos = data.geom_xpos[geom_id]
        mat = data.geom_xmat[geom_id].reshape(3, 3)
        world_corners = corners @ mat.T + pos
        return np.min(world_corners, axis=0), np.max(world_corners, axis=0)

# 使用 SAT (分離軸定理) 檢查兩個 OBB 是否重疊
def check_obb_overlap(obb1, obb2, margin=0.0):
    c1, axes1, e1 = obb1
    c2, axes2, e2 = obb2
    
    # 加上 margin (讓判定寬鬆一點點，避免穿模)
    e1 = e1 - margin 
    e2 = e2 - margin
    # 如果 margin 導致 extent 變負數，歸零
    e1 = np.maximum(e1, 0)
    e2 = np.maximum(e2, 0)

    # 兩中心連線向量
    T = c2 - c1
    
    # 需要測試的 15 個分離軸：
    # 3個 A 的軸, 3個 B 的軸, 9個 Cross Product
    
    # 輔助函式：投影測試
    # 如果 |T • L| > Σ |(Ai • L) * ei| + Σ |(Bi • L) * ei|，則分離 (無碰撞)
    def is_separated(axis):
        # 避免零向量
        if np.linalg.norm(axis) < 1e-6: return False
        
        # 投影半徑 A
        r_a = np.sum(np.abs(axes1.T @ axis) * e1)
        # 投影半徑 B
        r_b = np.sum(np.abs(axes2.T @ axis) * e2)
        # 中心距離投影
        dist = np.abs(np.dot(T, axis))
        
        return dist > (r_a + r_b)

    # 1. 測試 A 的 3 個軸
    for i in range(3):
        if is_separated(axes1[:, i]): return False # 分離，無碰撞

    # 2. 測試 B 的 3 個軸
    for i in range(3):
        if is_separated(axes2[:, i]): return False

    # 3. 測試 9 個外積軸 (Cross Products)
    for i in range(3):
        for j in range(3):
            axis = np.cross(axes1[:, i], axes2[:, j])
            if is_separated(axis): return False

    return True # 所有軸都重疊 -> 發生碰撞

# 改名為 calculator
collision_calc = CollisionCalculator()

# --- Scale Manager ---
class ScaleManager:
    def __init__(self):
        self.original_verts = {}; self.current_scales = {}
    def get_mesh_id(self, model, body_id):
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == body_id and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                return model.geom_dataid[i]
        return -1
    def cache_original_verts(self, model, mesh_id):
        if mesh_id not in self.original_verts and mesh_id != -1:
            vert_adr = model.mesh_vertadr[mesh_id]; vert_num = model.mesh_vertnum[mesh_id]
            self.original_verts[mesh_id] = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
            self.current_scales[mesh_id] = 1.0
    def apply_scale(self, model, ctx, body_id, scale):
        mesh_id = self.get_mesh_id(model, body_id)
        if mesh_id == -1: return
        self.cache_original_verts(model, mesh_id)
        vert_adr = model.mesh_vertadr[mesh_id]; vert_num = model.mesh_vertnum[mesh_id]
        model.mesh_vert[vert_adr : vert_adr + vert_num] = self.original_verts[mesh_id] * scale
        self.current_scales[mesh_id] = scale
        mujoco.mjr_uploadMesh(model, ctx, mesh_id)
    def get_current_scale(self, model, body_id):
        mesh_id = self.get_mesh_id(model, body_id)
        return self.current_scales.get(mesh_id, 1.0)

# --- History ---
class HistoryManager:
    def __init__(self, reload_callback, limit=20):
        self.undo_stack = []
        self.redo_stack = []
        self.limit = limit
        self.reload_callback = reload_callback 

    def push_state(self, xml_path, model, data):
        with open(xml_path, 'r', encoding='utf-8') as f: xml_content = f.read()
        state_snapshot = {'xml': xml_content, 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy()}
        self.undo_stack.append(state_snapshot)
        if len(self.undo_stack) > self.limit: self.undo_stack.pop(0)
        self.redo_stack.clear()
        print("[History] State saved.")

    def restore(self, snapshot, current_xml_path):
        with open(current_xml_path, 'w', encoding='utf-8') as f: f.write(snapshot['xml'])
        new_model, new_data = self.reload_callback(restore=False)
        try:
            if len(new_data.qpos) == len(snapshot['qpos']):
                new_data.qpos[:] = snapshot['qpos']
                new_data.qvel[:] = snapshot['qvel']
                mujoco.mj_forward(new_model, new_data)
        except Exception as e:
            print(f"[History] Restore physics error: {e}")

    def undo(self, current_xml_path, model, data):
        if not self.undo_stack: return
        self.redo_stack.append({'xml': open(current_xml_path, 'r', encoding='utf-8').read(), 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy()})
        self.restore(self.undo_stack.pop(), current_xml_path)
        print("[History] Undo performed.")

    def redo(self, current_xml_path, model, data):
        if not self.redo_stack: return
        self.undo_stack.append({'xml': open(current_xml_path, 'r', encoding='utf-8').read(), 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy()})
        self.restore(self.redo_stack.pop(), current_xml_path)
        print("[History] Redo performed.")

# --- Placement ---
class PlacementManager:
    def __init__(self):
        self.active_body_id = -1
        self.is_valid = False
        self.original_rgba = {}
        self.original_contype = {}
        self.original_conaffinity = {}
        self.initial_qpos = None 

    def start_placement(self, model, data, body_id):
        if self.active_body_id == body_id: return
        if self.active_body_id != -1: self.confirm_placement(model)

        self.active_body_id = body_id
        
        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr >= 0:
            qpos_adr = model.jnt_qposadr[jnt_adr]
            self.initial_qpos = data.qpos[qpos_adr : qpos_adr+7].copy()
        
        self.original_rgba.clear(); self.original_contype.clear(); self.original_conaffinity.clear()
        geoms = get_body_geoms(model, body_id)
        for gid in geoms:
            self.original_rgba[gid] = model.geom_rgba[gid].copy()
            self.original_contype[gid] = model.geom_contype[gid]
            self.original_conaffinity[gid] = model.geom_conaffinity[gid]
            model.geom_contype[gid] = 0; model.geom_conaffinity[gid] = 0

    def update(self, model, data):
        if self.active_body_id == -1: return
        is_overlapping = False
        my_geoms = get_body_geoms(model, self.active_body_id)
        
        # [修改] 改用 OBB 檢查
        for my_g in my_geoms:
            # 取得自己的 OBB
            my_obb = collision_calc.get_obb(model, data, my_g)
            
            for other_g in range(model.ngeom):
                if other_g in my_geoms or other_g == 0: continue
                if model.geom_contype[other_g] == 0: continue 
                
                # 取得對方的 OBB
                other_obb = collision_calc.get_obb(model, data, other_g)
                
                # [關鍵] 使用 SAT 檢查 OBB 重疊
                # margin 設定為負數 (例如 -0.005) 可以讓判定稍微寬鬆一點
                # 設定為 0.0 或正數則是非常嚴格
                if check_obb_overlap(my_obb, other_obb, margin=0.0):
                    is_overlapping = True
                    break
            if is_overlapping: break

        self.is_valid = not is_overlapping
        target_color = np.array([0.0, 1.0, 0.0, 0.5]) if self.is_valid else np.array([1.0, 0.0, 0.0, 0.5])
        for gid in my_geoms:
            model.geom_rgba[gid] = target_color

    def _cleanup_visuals(self, model):
        geoms = get_body_geoms(model, self.active_body_id)
        for gid in geoms:
            if gid in self.original_rgba: model.geom_rgba[gid] = self.original_rgba[gid]
            if gid in self.original_contype: model.geom_contype[gid] = self.original_contype[gid]
            if gid in self.original_conaffinity: model.geom_conaffinity[gid] = self.original_conaffinity[gid]
        self.original_rgba.clear()

    def confirm_placement(self, model):
        if self.active_body_id == -1: return
        self._cleanup_visuals(model)
        self.active_body_id = -1
        self.initial_qpos = None

    def revert_placement(self, model, data):
        if self.active_body_id != -1 and self.initial_qpos is not None:
            jnt_adr = model.body_jntadr[self.active_body_id]
            if jnt_adr >= 0:
                qpos_adr = model.jnt_qposadr[jnt_adr]
                data.qpos[qpos_adr : qpos_adr+7] = self.initial_qpos.copy()
            
            self._cleanup_visuals(model)
            self.active_body_id = -1
            self.initial_qpos = None
            return True
        return False
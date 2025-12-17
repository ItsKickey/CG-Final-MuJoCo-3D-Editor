# src/managers.py
import numpy as np
import mujoco
from src.utils import get_body_geoms, save_simulation_state, restore_simulation_state

# --- AABB ---
class AABBCalculator:
    def __init__(self): self.local_bounds_cache = {}
    
    def get_local_bounds(self, model, geom_id):
        if geom_id in self.local_bounds_cache: return self.local_bounds_cache[geom_id]
        g_type = model.geom_type[geom_id]
        if g_type == mujoco.mjtGeom.mjGEOM_MESH:
            dataid = model.geom_dataid[geom_id]
            if dataid != -1:
                vert_adr = model.mesh_vertadr[dataid]; vert_num = model.mesh_vertnum[dataid]
                verts = model.mesh_vert[vert_adr : vert_adr + vert_num]
                if len(verts) > 0:
                    min_v = np.min(verts, axis=0); max_v = np.max(verts, axis=0)
                    self.local_bounds_cache[geom_id] = (min_v, max_v)
                    return min_v, max_v
        size = model.geom_size[geom_id]
        if g_type == mujoco.mjtGeom.mjGEOM_BOX: min_v = -size[:3]; max_v = size[:3]
        elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE: r = size[0]; min_v = np.array([-r]*3); max_v = np.array([r]*3)
        else: r = model.geom_rbound[geom_id]; min_v = np.array([-r]*3); max_v = np.array([r]*3)
        self.local_bounds_cache[geom_id] = (min_v, max_v)
        return min_v, max_v

    def get_world_aabb(self, model, data, geom_id):
        min_local, max_local = self.get_local_bounds(model, geom_id)
        corners = np.array([
            [min_local[0], min_local[1], min_local[2]], [min_local[0], min_local[1], max_local[2]],
            [min_local[0], max_local[1], min_local[2]], [min_local[0], max_local[1], max_local[2]],
            [max_local[0], min_local[1], min_local[2]], [max_local[0], min_local[1], max_local[2]],
            [max_local[0], max_local[1], min_local[2]], [max_local[0], max_local[1], max_local[2]],
        ])
        pos = data.geom_xpos[geom_id]; mat = data.geom_xmat[geom_id].reshape(3, 3)
        world_corners = corners @ mat.T + pos
        return np.min(world_corners, axis=0), np.max(world_corners, axis=0)

def check_aabb_overlap(min1, max1, min2, max2, margin=0.0):
    if max1[0] < min2[0] + margin or min1[0] > max2[0] - margin: return False
    if max1[1] < min2[1] + margin or min1[1] > max2[1] - margin: return False
    if max1[2] < min2[2] + margin or min1[2] > max2[2] - margin: return False
    return True

# 建立全域實例供 PlacementManager 使用
aabb_calc = AABBCalculator()

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
        self.reload_callback = reload_callback # 這是 main.py 裡的 load_model

    def push_state(self, xml_path, model, data):
        with open(xml_path, 'r', encoding='utf-8') as f: xml_content = f.read()
        state_snapshot = {'xml': xml_content, 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy()}
        self.undo_stack.append(state_snapshot)
        if len(self.undo_stack) > self.limit: self.undo_stack.pop(0)
        self.redo_stack.clear()
        print("[History] State saved.")

    def restore(self, snapshot, current_xml_path):
        with open(current_xml_path, 'w', encoding='utf-8') as f: f.write(snapshot['xml'])
        
        # 呼叫主程式傳進來的 reload 函數 (相當於 load_model(restore=False))
        # 這裡我們需要一個機制來取回新的 state.model 和 state.data
        # 為了簡單，我們假設 callback 會更新全域狀態，我們這裡只負責觸發
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
        self.active_body_id = -1; self.is_valid = False
        self.original_rgba = {}; self.original_contype = {}; self.original_conaffinity = {}

    def start_placement(self, model, body_id):
        if self.active_body_id != -1: self.confirm_placement(model)
        self.active_body_id = body_id
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
        for my_g in my_geoms:
            my_min, my_max = aabb_calc.get_world_aabb(model, data, my_g)
            for other_g in range(model.ngeom):
                if other_g in my_geoms or other_g == 0: continue
                if model.geom_contype[other_g] == 0: continue 
                other_min, other_max = aabb_calc.get_world_aabb(model, data, other_g)
                if check_aabb_overlap(my_min, my_max, other_min, other_max, margin=0.01):
                    is_overlapping = True; break
            if is_overlapping: break
        self.is_valid = not is_overlapping
        target_color = np.array([0.0, 1.0, 0.0, 0.5]) if self.is_valid else np.array([1.0, 0.0, 0.0, 0.5])
        for gid in my_geoms: model.geom_rgba[gid] = target_color

    def confirm_placement(self, model):
        if self.active_body_id == -1: return
        geoms = get_body_geoms(model, self.active_body_id)
        for gid in geoms:
            if gid in self.original_rgba: model.geom_rgba[gid] = self.original_rgba[gid]
            if gid in self.original_contype: model.geom_contype[gid] = self.original_contype[gid]
            if gid in self.original_conaffinity: model.geom_conaffinity[gid] = self.original_conaffinity[gid]
        self.active_body_id = -1; self.original_rgba.clear()
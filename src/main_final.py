# src/main_final.py
import mujoco
import glfw
import numpy as np
import os
import sys
import math
import shutil
import tkinter as tk
from tkinter import filedialog

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 使用 loader V18
from src.loader import load_scene_with_object, delete_body_from_scene, update_body_xml, LAST_IMPORTED_BODY_NAME
from src.importer import convert_obj_with_obj2mjcf

BASE_XML_PATH = "scene/main_scene.xml"
CURRENT_SCENE_XML = "scene/current_scene.xml"
GRID_SIZE = 0.5

if not os.path.exists("scene"): os.makedirs("scene")
if os.path.exists(BASE_XML_PATH):
    shutil.copy(BASE_XML_PATH, CURRENT_SCENE_XML)

active_xml_path = CURRENT_SCENE_XML

# --- Math ---
def euler2quat(r, p, y):
    r, p, y = np.radians([r, p, y])
    cr, sr = np.cos(r*0.5), np.sin(r*0.5)
    cp, sp = np.cos(p*0.5), np.sin(p*0.5)
    cy, sy = np.cos(y*0.5), np.sin(y*0.5)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

def quat2euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2*(w*x + y*z); cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(sinp) if abs(sinp) < 1 else np.copysign(np.pi/2, sinp)
    siny_cosp = 2*(w*z + x*y); cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])

def get_body_geoms(model, body_id):
    geoms = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == body_id: geoms.append(i)
    return geoms

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

aabb_calc = AABBCalculator()

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

scale_mgr = ScaleManager()

# --- History ---
class HistoryManager:
    def __init__(self, limit=20):
        self.undo_stack = []
        self.redo_stack = []
        self.limit = limit

    def push_state(self, xml_path, model, data):
        with open(xml_path, 'r', encoding='utf-8') as f: xml_content = f.read()
        state_snapshot = {'xml': xml_content, 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy()}
        self.undo_stack.append(state_snapshot)
        if len(self.undo_stack) > self.limit: self.undo_stack.pop(0)
        self.redo_stack.clear()
        print("[History] State saved.")

    def restore(self, snapshot, current_xml_path):
        with open(current_xml_path, 'w', encoding='utf-8') as f: f.write(snapshot['xml'])
        load_model(restore=False) # 載入 XML 的狀態，不要保留當前記憶體的狀態
        try:
            if len(state.data.qpos) == len(snapshot['qpos']):
                state.data.qpos[:] = snapshot['qpos']
                state.data.qvel[:] = snapshot['qvel']
                mujoco.mj_forward(state.model, state.data)
        except: pass

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

history = HistoryManager()

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

pm = PlacementManager()

# --- GUI ---
class ControlPanel:
    def __init__(self, load_cb, open_cb, rot_cb, confirm_cb, delete_cb, save_cb, undo_cb, redo_cb):
        self.root = tk.Tk()
        self.root.title("Control Panel")
        self.root.geometry("320x850")
        self.root.configure(bg="#f0f0f0")
        self.root.attributes("-topmost", True)

        tk.Label(self.root, text="Furniture Placer", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        # File
        frame_file = tk.LabelFrame(self.root, text="File", padx=10, pady=10, bg="#f0f0f0")
        frame_file.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_file, text="📂 Open Scene...", command=open_cb, bg="#e1e1e1").pack(fill="x", pady=2)
        tk.Button(frame_file, text="💾 Save Scene As...", command=save_cb, bg="#ccf").pack(fill="x", pady=2)
        
        # Add Object
        frame_add = tk.LabelFrame(self.root, text="Add", padx=10, pady=10, bg="#f0f0f0")
        frame_add.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_add, text="➕ Add OBJ (I)", command=load_cb, bg="#e1e1e1", height=2).pack(fill="x")

        # History
        frame_hist = tk.LabelFrame(self.root, text="History", padx=10, pady=10, bg="#f0f0f0")
        frame_hist.pack(fill="x", padx=10, pady=5)
        btn_undo = tk.Button(frame_hist, text="↩ Undo (Ctrl+Z)", command=undo_cb, width=15)
        btn_undo.pack(side="left", padx=5)
        btn_redo = tk.Button(frame_hist, text="↪ Redo (Ctrl+Y)", command=redo_cb, width=15)
        btn_redo.pack(side="right", padx=5)

        # Placement
        self.frame_place = tk.LabelFrame(self.root, text="Placement", padx=10, pady=10, bg="#f0f0f0")
        self.frame_place.pack(fill="x", padx=10, pady=5)
        self.lbl_validity = tk.Label(self.frame_place, text="Waiting...", font=("Arial", 12, "bold"), bg="#ddd", width=15)
        self.lbl_validity.pack(pady=5)
        self.btn_confirm = tk.Button(self.frame_place, text="✅ Confirm (Enter)", command=confirm_cb, bg="#8f8", state="disabled", height=2)
        self.btn_confirm.pack(fill="x", pady=5)
        self.btn_delete = tk.Button(self.frame_place, text="🗑️ Delete (Del)", command=delete_cb, bg="#f88", state="disabled", height=2)
        self.btn_delete.pack(fill="x", pady=5)

        # Transform
        frame_trans = tk.LabelFrame(self.root, text="Transform", padx=10, pady=10, bg="#f0f0f0")
        frame_trans.pack(fill="x", padx=10, pady=5)
        self.var_z = tk.DoubleVar(); self.var_scale = tk.DoubleVar(value=1.0)
        self.var_roll = tk.DoubleVar(); self.var_pitch = tk.DoubleVar(); self.var_yaw = tk.DoubleVar()
        self.is_updating = False

        def on_change(_):
            if not self.is_updating:
                rot_cb(self.var_z.get(), self.var_scale.get(), self.var_roll.get(), self.var_pitch.get(), self.var_yaw.get())

        tk.Label(frame_trans, text="Height (Z)", bg="#f0f0f0", fg="blue").pack(anchor="w")
        self.s_z = tk.Scale(frame_trans, variable=self.var_z, from_=0.0, to=3.0, resolution=0.05, orient="horizontal", command=on_change, bg="#f0f0f0")
        self.s_z.pack(fill="x")
        tk.Label(frame_trans, text="Scale", bg="#f0f0f0", fg="red").pack(anchor="w")
        self.s_scale = tk.Scale(frame_trans, variable=self.var_scale, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", command=on_change, bg="#f0f0f0")
        self.s_scale.pack(fill="x")
        for label, var in [("Roll (X)", self.var_roll), ("Pitch (Y)", self.var_pitch), ("Yaw (Z)", self.var_yaw)]:
            tk.Label(frame_trans, text=label, bg="#f0f0f0").pack(anchor="w")
            tk.Scale(frame_trans, variable=var, from_=-180, to=180, orient="horizontal", command=on_change, bg="#f0f0f0").pack(fill="x")

        self.lbl_status = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor="w")
        self.lbl_status.pack(side="bottom", fill="x")

    def update_gui_state(self, is_placing, is_valid, has_selection):
        if is_placing:
            if is_valid:
                self.lbl_validity.config(text="VALID", bg="#8f8", fg="black")
                self.btn_confirm.config(state="normal", bg="#8f8")
            else:
                self.lbl_validity.config(text="OVERLAP", bg="#f88", fg="white")
                self.btn_confirm.config(state="disabled", bg="#ddd")
        else:
            self.lbl_validity.config(text="No Selection", bg="#ddd", fg="#555")
            self.btn_confirm.config(state="disabled", bg="#ddd")
        
        self.btn_delete.config(state="normal" if has_selection else "disabled", bg="#f88" if has_selection else "#ddd")

    def update(self):
        try: self.root.update_idletasks(); self.root.update()
        except: pass

    def set_status(self, msg): self.lbl_status.config(text=msg)
    
    def set_values(self, z, s, r, p, y):
        self.is_updating = True
        self.var_z.set(z); self.var_scale.set(s)
        self.var_roll.set(r); self.var_pitch.set(p); self.var_yaw.set(y)
        self.is_updating = False

# --- State ---
class EditorState:
    def __init__(self):
        self.scale = 1.0
        self.last_mouse_x = 0; self.last_mouse_y = 0
        self.is_dragging = False; self.button_left_pressed = False; self.shift_pressed = False
        self.window = None; self.model = None; self.data = None
        self.cam = mujoco.MjvCamera(); self.opt = mujoco.MjvOption()
        self.scn = None; self.ctx = None
        self.selected_body_id = -1; self.selected_qpos_adr = -1
        self.gui = None
        self.current_z_height = 0.0

state = EditorState()

# --- Logic ---
def save_simulation_state(model, data):
    if model is None or data is None: return {}
    saved = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_adr = model.jnt_qposadr[i]; qvel_adr = model.jnt_dofadr[i]
            jnt_type = model.jnt_type[i]
            q_len = 7 if jnt_type == mujoco.mjtJoint.mjJNT_FREE else 1
            v_len = 6 if jnt_type == mujoco.mjtJoint.mjJNT_FREE else 1
            saved[name] = {"qpos": data.qpos[qpos_adr:qpos_adr+q_len].copy(), 
                           "qvel": data.qvel[qvel_adr:qvel_adr+v_len].copy()}
    return saved

def restore_simulation_state(model, data, saved_state):
    if not saved_state: return
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name in saved_state:
            vals = saved_state[name]
            qpos_adr = model.jnt_qposadr[i]; qvel_adr = model.jnt_dofadr[i]
            q_len = len(vals["qpos"]); v_len = len(vals["qvel"])
            data.qpos[qpos_adr:qpos_adr+q_len] = vals["qpos"]
            data.qvel[qvel_adr:qvel_adr+v_len] = vals["qvel"]
    mujoco.mj_forward(model, data)

def load_model(restore=True):
    print(f"[System] Reloading model... (Restore={restore})")
    
    # 只有在熱重載時才保存當前狀態
    old_state = save_simulation_state(state.model, state.data) if restore else None
    
    try:
        state.model = mujoco.MjModel.from_xml_path(active_xml_path)
        state.data = mujoco.MjData(state.model)
        
        # 如果是熱重載，嘗試還原物體位置
        if restore and old_state:
            restore_simulation_state(state.model, state.data, old_state)
            
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        state.selected_body_id = -1
        state.selected_qpos_adr = -1
    except Exception as e: print(f"Error loading: {e}")

# [Action] Open Scene
def open_scene():
    path = filedialog.askopenfilename(filetypes=[("XML", "*.xml")])
    if not path: return
    
    # 複製選中的檔案到 current_scene.xml (作為工作檔)
    shutil.copy(path, CURRENT_SCENE_XML)
    
    # 清空 Undo 歷史，因為這是新場景
    history.undo_stack.clear()
    history.redo_stack.clear()
    
    # 重置編輯器狀態
    pm.active_body_id = -1
    state.selected_body_id = -1
    state.selected_qpos_adr = -1
    
    # 載入且「不」還原舊狀態 (使用檔案裡的座標)
    load_model(restore=False)
    
    if state.gui: state.gui.set_status(f"Opened: {os.path.basename(path)}")

# [Action] Import
def import_obj_workflow(obj_path=None):
    if not obj_path:
        obj_path = filedialog.askopenfilename(title="Select OBJ", filetypes=[("OBJ", "*.obj")])
    if not obj_path: return

    try:
        if state.gui: state.gui.set_status("Importing...")
        if pm.active_body_id != -1: confirm_current_placement()

        history.push_state(active_xml_path, state.model, state.data)

        mjcf_path = convert_obj_with_obj2mjcf(obj_path)
        spawn_h = state.current_z_height if state.current_z_height > 0 else 0.5
        
        # 匯入時保留其他物體位置 (Implicitly handled by load_scene_with_object merging XML)
        state.model = load_scene_with_object(active_xml_path, str(mjcf_path), spawn_height=spawn_h, save_merged_xml=active_xml_path)
        
        # 重載並還原其他物體狀態
        # 注意：這裡我們手動操作，不呼叫 load_model 以免重複讀檔
        # 但為了簡單，我們可以依賴 load_model(restore=True) 的行為
        # 不過因為 state.model 已經是新的了，我們需要把舊狀態還原進去
        old_qpos = save_simulation_state(state.model, state.data) # Dummy save? No, we need old data.
        # 其實 loader 已經把舊物體的 pos 寫在 XML 裡了，所以直接 init 新 data 應該就會在正確位置！
        # 唯一例外是如果舊物體正在掉落中還沒落地，XML 裡的 pos 可能是舊的。
        # 但既然我們有 "Confirm" 機制，所有靜止物體的位置都已寫入 XML。
        # 只有 "正在掉落" 的物體會被重置。這在編輯器邏輯裡是可以接受的。
        
        state.data = mujoco.MjData(state.model)
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 嘗試還原 (如果之前有保存狀態的話)
        if len(history.undo_stack) > 0:
             # 從 Undo stack 拿最近的狀態來還原非靜止物體
             prev_snapshot = history.undo_stack[-1]
             # restore_simulation_state... (略，依賴 XML 其實最穩)

        if LAST_IMPORTED_BODY_NAME:
            bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, LAST_IMPORTED_BODY_NAME)
            if bid >= 0:
                select_object_by_id(bid)
                if state.gui: state.gui.set_status(f"Imported: {os.path.basename(obj_path)}")
    except Exception as e:
        print(e); import traceback; traceback.print_exc()

# [Action] Delete
def delete_selected_object():
    if state.selected_body_id == -1: return
    body_name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, state.selected_body_id)
    if not body_name: return
    
    history.push_state(active_xml_path, state.model, state.data)
    
    if delete_body_from_scene(active_xml_path, body_name):
        load_model(restore=True) # 刪除後保持其他物體位置
        pm.active_body_id = -1
        state.selected_body_id = -1
        if state.gui: state.gui.set_status(f"Deleted: {body_name}")

# [Action] Confirm
def confirm_current_placement():
    if pm.active_body_id != -1 and pm.is_valid:
        history.push_state(active_xml_path, state.model, state.data)
        
        body_name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, pm.active_body_id)
        if body_name:
            jnt_adr = state.model.body_jntadr[pm.active_body_id]
            qpos_adr = state.model.jnt_qposadr[jnt_adr]
            pos = state.data.qpos[qpos_adr : qpos_adr+3]
            quat = state.data.qpos[qpos_adr+3 : qpos_adr+7]
            update_body_xml(active_xml_path, body_name, pos, quat)
        
        pm.confirm_placement(state.model)
        state.selected_body_id = -1
        state.selected_qpos_adr = -1
        state.gui.set_status("Placed successfully.")

# [Action] Save Scene
def save_scene_as():
    path = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML", "*.xml")])
    if path:
        shutil.copy(active_xml_path, path)
        if state.gui: state.gui.set_status(f"Saved to {os.path.basename(path)}")

# [Action] Undo/Redo
def perform_undo():
    history.undo(active_xml_path, state.model, state.data)
    # Undo 後必須重建渲染環境，因為 XML 結構可能變了 (例如刪除物體後復原)
    state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
    state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    pm.active_body_id = -1

def perform_redo():
    history.redo(active_xml_path, state.model, state.data)
    state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
    state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    pm.active_body_id = -1

# ... (select_object_by_id, update_transform_from_gui, pick_object, raycast, etc. 同前) ...
# (請務必保留上一版的所有 helper functions)

def select_object_by_id(body_id):
    if pm.active_body_id != -1 and pm.active_body_id != body_id:
        confirm_current_placement()

    state.selected_body_id = body_id
    jntadr = state.model.body_jntadr[body_id]
    if jntadr >= 0:
        state.selected_qpos_adr = state.model.jnt_qposadr[jntadr]
        z = state.data.qpos[state.selected_qpos_adr + 2]
        quat = state.data.qpos[state.selected_qpos_adr+3 : state.selected_qpos_adr+7]
        r, p, y = quat2euler(quat)
        s = scale_mgr.get_current_scale(state.model, body_id)
        state.current_z_height = z 
        if state.gui: state.gui.set_values(z, s, r, p, y)
        pm.start_placement(state.model, body_id)
    else: state.selected_qpos_adr = -1

def update_transform_from_gui(z, s, r, p, y):
    if state.selected_qpos_adr < 0: return
    state.current_z_height = z
    state.data.qpos[state.selected_qpos_adr + 2] = z
    state.data.qpos[state.selected_qpos_adr+3 : state.selected_qpos_adr+7] = euler2quat(r, p, y)
    scale_mgr.apply_scale(state.model, state.ctx, state.selected_body_id, s)
    mujoco.mj_forward(state.model, state.data)

def pick_object(window, xpos, ypos):
    width, height = glfw.get_framebuffer_size(window)
    selpnt = np.zeros(3); selgeom = np.zeros(1, dtype=np.int32); selflex = np.zeros(1, dtype=np.int32); selskin = np.zeros(1, dtype=np.int32)
    mujoco.mjv_select(state.model, state.data, state.opt, width/height, xpos/width, (height-ypos)/height, state.scn, selpnt, selgeom, selflex, selskin)
    
    geom_id = selgeom[0]
    if geom_id < 0:
        if pm.active_body_id != -1:
            if pm.is_valid: confirm_current_placement()
            else: 
                if state.gui: state.gui.set_status("Cannot place here!")
        return
    
    body_id = state.model.geom_bodyid[geom_id]
    if body_id > 0:
        select_object_by_id(body_id)
        if state.gui: state.gui.set_status(f"Moving ID: {body_id}")

def raycast_to_ground(window, xpos, ypos, cam):
    width, height = glfw.get_framebuffer_size(window)
    ndc_x = (2*xpos/width) - 1; ndc_y = 1 - (2*ypos/height)
    aspect = width/height; tan_half = math.tan(math.radians(45/2))
    ray_cam = np.array([ndc_x*tan_half*aspect, ndc_y*tan_half, -1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    az = math.radians(cam.azimuth - 90); el = math.radians(cam.elevation)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    R = Rz @ Rx
    ray_world = R @ ray_cam
    cam_pos = cam.lookat + R @ np.array([0, 0, cam.distance])
    if abs(ray_world[2]) < 1e-6: return None
    t = -cam_pos[2] / ray_world[2]
    return cam_pos + t * ray_world if t > 0 else None

def snap_to_grid(val): return round(val / GRID_SIZE) * GRID_SIZE

def mouse_button_callback(window, button, action, mods):
    state.button_left_pressed = (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS)
    state.shift_pressed = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            if state.shift_pressed: state.is_dragging = True
            else: pick_object(window, state.last_mouse_x, state.last_mouse_y)
        elif action == glfw.RELEASE: state.is_dragging = False

def cursor_pos_callback(window, xpos, ypos):
    dx = xpos - state.last_mouse_x; dy = ypos - state.last_mouse_y
    state.last_mouse_x = xpos; state.last_mouse_y = ypos
    if state.is_dragging and state.selected_qpos_adr != -1:
        p = raycast_to_ground(window, xpos, ypos, state.cam)
        if p is not None:
            state.data.qpos[state.selected_qpos_adr] = snap_to_grid(p[0])
            state.data.qpos[state.selected_qpos_adr+1] = snap_to_grid(p[1])
            state.data.qpos[state.selected_qpos_adr+2] = state.current_z_height
            mujoco.mj_forward(state.model, state.data) 
            select_object_by_id(state.selected_body_id) 
    elif state.button_left_pressed and not state.shift_pressed:
        mujoco.mjv_moveCamera(state.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx/500, dy/500, state.scn, state.cam)

def scroll_callback(window, xoffset, yoffset):
    state.cam.distance -= yoffset * 0.5
    if state.cam.distance < 0.1: state.cam.distance = 0.1

def key_callback(window, key, scancode, action, mods):
    if action == glfw.PRESS:
        if key == glfw.KEY_I: import_obj_workflow()
        elif key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_ENTER: confirm_current_placement()
        elif key == glfw.KEY_DELETE: delete_selected_object()
        elif key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL): perform_undo()
        elif key == glfw.KEY_Y and (mods & glfw.MOD_CONTROL): perform_redo()

def main():
    if not glfw.init(): return
    window = glfw.create_window(1200, 900, "Final Project Editor", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    state.window = window

    # 更新 GUI 建構子參數，加入 open_cb
    state.gui = ControlPanel(import_obj_workflow, open_scene, update_transform_from_gui, confirm_current_placement, delete_selected_object, save_scene_as, perform_undo, perform_redo)

    # Initial Load (Clean state)
    try:
        obj_a_xml = "Assets/obj/basemodule_A/basemodule_A.xml"
        state.model = load_scene_with_object(active_xml_path, obj_a_xml, spawn_height=0.0, save_merged_xml=active_xml_path)
        state.data = mujoco.MjData(state.model)
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    except: pass

    state.cam.azimuth = 90; state.cam.elevation = -45; state.cam.distance = 10
    state.cam.lookat = np.array([0.0, 0.0, 0.0])

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    while not glfw.window_should_close(window):
        state.gui.update()
        sim_start = state.data.time
        while state.data.time - sim_start < 1.0/60.0:
            pm.update(state.model, state.data)
            state.gui.update_gui_state(pm.active_body_id != -1, pm.is_valid, state.selected_body_id != -1)

            state.data.qfrc_applied[:] = 0 
            if pm.active_body_id != -1:
                for i in range(state.model.nbody):
                    if i == 0: continue
                    jnt_adr = state.model.body_jntadr[i]
                    if jnt_adr >= 0 and state.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                        dof_adr = state.model.jnt_dofadr[jnt_adr]
                        state.data.qvel[dof_adr : dof_adr+6] = 0
                        state.data.qfrc_applied[dof_adr : dof_adr+6] = state.data.qfrc_bias[dof_adr : dof_adr+6]
            
            # Void Safety
            for i in range(state.model.nbody):
                jnt_adr = state.model.body_jntadr[i]
                if jnt_adr >= 0 and state.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_adr = state.model.jnt_qposadr[jnt_adr]
                    if state.data.qpos[qpos_adr+2] < -5.0:
                        state.data.qpos[qpos_adr:qpos_adr+3] = [0,0,5]
                        state.data.qvel[state.model.jnt_dofadr[jnt_adr]:state.model.jnt_dofadr[jnt_adr]+6] = 0

            mujoco.mj_step(state.model, state.data)

        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(state.model, state.data, state.opt, None, state.cam, mujoco.mjtCatBit.mjCAT_ALL.value, state.scn)
        
        if state.selected_body_id != -1:
            for i in range(state.scn.ngeom):
                g = state.scn.geoms[i]
                bid = state.model.geom_bodyid[g.objid]
                if bid == state.selected_body_id: g.emission += 0.3

        mujoco.mjr_render(viewport, state.scn, state.ctx)
        glfw.swap_buffers(window); glfw.poll_events()
    
    state.gui.root.destroy()
    glfw.terminate()

if __name__ == "__main__":
    main()
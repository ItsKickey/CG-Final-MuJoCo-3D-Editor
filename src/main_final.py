# src/main_final.py
import os
os.environ["MUJOCO_GL"] = "glfw"
import mujoco
import glfw
import numpy as np
import sys
import math
import shutil
import time
from tkinter import filedialog, simpledialog

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用模組方式匯入，以便讀取全域變數
import src.loader 
from src.loader import (
    load_scene_with_object, delete_body_from_scene, update_body_xml, 
    add_light_to_scene, update_light_xml, change_floor_texture, 
    batch_update_bodies_xml
)
from src.importer import convert_obj_with_obj2mjcf
from src.utils import euler2quat, quat2euler, save_simulation_state, restore_simulation_state
from src.managers import ScaleManager, PlacementManager, HistoryManager
from src.gui import ImGuiPanel
from src.initializer import initialize_project
from src.export import export_project_to_zip
from src.logger import setup_logging

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

BASE_XML_PATH, CURRENT_SCENE_XML = initialize_project()
GRID_SIZE = 0.5
active_xml_path = CURRENT_SCENE_XML

# --- Global State ---
class EditorState:
    def __init__(self):
        self.scale = 1.0
        self.last_mouse_x = 0; self.last_mouse_y = 0
        self.is_dragging = False; 
        
        self.button_left_pressed = False; 
        self.button_right_pressed = False 
        self.shift_pressed = False
        
        self.window = None; self.model = None; self.data = None
        self.cam = mujoco.MjvCamera() 
        self.opt = mujoco.MjvOption()
        self.scn = None; self.ctx = None
        self.selected_body_id = -1; self.selected_qpos_adr = -1
        self.gui = None; self.listbox_body_ids = []; self.pending_tasks = []
        self.last_auto_save_time = 0 
        self.object_names = []; self.listbox_index = -1
        self.current_scale = 1.0; self.current_z_height = 0.0
        self.current_roll = 0.0; self.current_pitch = 0.0; self.current_yaw = 0.0
        self.current_rgb = (1.0, 1.0, 1.0)
        self.is_placing = False; self.is_valid = False
        self.is_light_selected = False
        
        # [移除] self.paused = False (不需要暫停了)
        self.target_fps = 60.0

    def defer(self, func, *args):
        self.pending_tasks.append(lambda: func(*args))

state = EditorState()
scale_mgr = ScaleManager()
pm = PlacementManager()

# --- Helper Functions ---
def refresh_object_list_ui():
    if not state.model: return
    names = []; state.listbox_body_ids = []
    for i in range(state.model.nbody):
        if i == 0: continue
        name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name: name = f"Body {i}"
        names.append(name); state.listbox_body_ids.append(i)
    state.object_names = names

def on_gui_list_select(index):
    if 0 <= index < len(state.listbox_body_ids):
        body_id = state.listbox_body_ids[index]
        select_object_by_id(body_id)

def load_model(restore=True):
    print(f"[System] Reloading model... (Restore={restore})")
    old_state = save_simulation_state(state.model, state.data) if restore else None
    try:
        state.model = mujoco.MjModel.from_xml_path(active_xml_path)
        state.data = mujoco.MjData(state.model)
        if restore and old_state: restore_simulation_state(state.model, state.data, old_state)
        mujoco.mj_forward(state.model, state.data) 
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        state.selected_body_id = -1; state.selected_qpos_adr = -1; state.is_light_selected = False
        refresh_object_list_ui()
        return state.model, state.data
    except Exception as e: 
        print(f"Error loading: {e}"); return None, None

history = HistoryManager(reload_callback=load_model)

def get_light_idx_for_body(body_id):
    if state.model is None: return -1
    for i in range(state.model.nlight):
        if state.model.light_bodyid[i] == body_id: return i
    return -1

def cancel_active_object():
    if pm.active_body_id != -1:
        print(f"[Auto-Revert] Action without Confirm. Reverting Body {pm.active_body_id}...")
        pm.revert_placement(state.model, state.data)
        state.selected_body_id = -1; state.selected_qpos_adr = -1; state.is_light_selected = False
        if state.gui: 
            state.gui.set_status("Edit Cancelled (Reverted)")
            state.listbox_index = -1 
        return True
    return False

# --- Logic Functions ---
def _export_project_logic():
    scene_name = simpledialog.askstring("Export Project", "Enter Scene Name:\n(Files will be saved to outputfile/Name.zip)")
    if not scene_name: return 
    valid_name = "".join(c for c in scene_name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not valid_name: 
        if state.gui: state.gui.set_status("Invalid Name!")
        return
    try:
        if state.gui: state.gui.set_status("Exporting Zip...")
        batch_update_bodies_xml(active_xml_path, state.model, state.data)
        zip_path = export_project_to_zip(active_xml_path, valid_name)
        if state.gui: state.gui.set_status(f"Exported: {os.path.basename(zip_path)}")
    except Exception as e:
        print(f"Export Error: {e}")
        if state.gui: state.gui.set_status("Export Failed!")

def _change_floor_workflow_logic():
    img_path = filedialog.askopenfilename(title="Select Floor Image", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not img_path: return
    try:
        if state.gui: state.gui.set_status("Updating Floor...")
        history.push_state(active_xml_path, state.model, state.data)
        if change_floor_texture(active_xml_path, img_path):
            load_model(restore=True)
            if state.gui: state.gui.set_status(f"Floor updated: {os.path.basename(img_path)}")
    except Exception as e: print(e)

def _open_scene_logic():
    path = filedialog.askopenfilename(filetypes=[("XML", "*.xml")])
    if not path: return
    shutil.copy(path, CURRENT_SCENE_XML)
    history.undo_stack.clear(); history.redo_stack.clear()
    pm.active_body_id = -1; state.selected_body_id = -1
    load_model(restore=False)
    if state.gui: state.gui.set_status(f"Opened: {os.path.basename(path)}")

def _import_obj_workflow_logic(obj_path=None):
    if not obj_path:
        obj_path = filedialog.askopenfilename(title="Select OBJ", filetypes=[("OBJ", "*.obj")])
    if not obj_path: return
    try:
        if state.gui: state.gui.set_status("Importing...")
        cancel_active_object()
        batch_update_bodies_xml(active_xml_path, state.model, state.data)
        history.push_state(active_xml_path, state.model, state.data)
        mjcf_path = convert_obj_with_obj2mjcf(obj_path)
        spawn_h = state.current_z_height if state.current_z_height > 0 else 0.5
        state.model = load_scene_with_object(active_xml_path, str(mjcf_path), spawn_height=spawn_h, save_merged_xml=active_xml_path)
        state.data = mujoco.MjData(state.model)
        mujoco.mj_forward(state.model, state.data)
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        refresh_object_list_ui()
        
        target_bid = -1
        # 自動選取剛匯入的物件
        if src.loader.LAST_IMPORTED_BODY_NAME:
            target_bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, src.loader.LAST_IMPORTED_BODY_NAME)
        
        if target_bid >= 0:
            select_object_by_id(target_bid) # 這會啟動 PlacementManager，自動關閉碰撞
            
            if state.gui: state.gui.set_status(f"Imported: {os.path.basename(obj_path)}")
            # 鏡頭看過去
            state.cam.lookat = state.data.body_xpos[target_bid].copy()
            state.cam.distance = 5.0
            pm.update(state.model, state.data)

    except Exception as e:
        print(f"Import Error: {e}")
        if state.gui: state.gui.set_status("Import Failed!")
        load_model(restore=False)

def _add_light_workflow_logic():
    try:
        if state.gui: state.gui.set_status("Adding Light...")
        cancel_active_object()
        batch_update_bodies_xml(active_xml_path, state.model, state.data)
        history.push_state(active_xml_path, state.model, state.data)
        spawn_h = state.current_z_height if state.current_z_height > 0 else 3.0
        if add_light_to_scene(active_xml_path, spawn_pos=f"0 0 {spawn_h}"):
            load_model(restore=True)
            # 新增燈光後也自動選取
            if src.loader.LAST_IMPORTED_BODY_NAME:
                bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, src.loader.LAST_IMPORTED_BODY_NAME)
                if bid >= 0:
                    select_object_by_id(bid)
                    if state.gui: state.gui.set_status("Added Point Light")
    except Exception as e: print(e)

def _delete_selected_object_logic():
    if state.selected_body_id == -1: return
    body_name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, state.selected_body_id)
    if not body_name: return
    history.push_state(active_xml_path, state.model, state.data)
    if delete_body_from_scene(active_xml_path, body_name):
        load_model(restore=True)
        pm.active_body_id = -1; state.selected_body_id = -1
        if state.gui: state.gui.set_status(f"Deleted: {body_name}")

def _perform_undo_logic():
    history.undo(active_xml_path, state.model, state.data)
    state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
    state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    pm.active_body_id = -1; refresh_object_list_ui()

def _perform_redo_logic():
    history.redo(active_xml_path, state.model, state.data)
    state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
    state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    pm.active_body_id = -1; refresh_object_list_ui()

def _confirm_current_placement_logic():
    if pm.active_body_id != -1 and pm.is_valid:
        history.push_state(active_xml_path, state.model, state.data)
        body_name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, pm.active_body_id)
        if body_name:
            jnt_adr = state.model.body_jntadr[pm.active_body_id]
            qpos_adr = state.model.jnt_qposadr[jnt_adr]
            pos = state.data.qpos[qpos_adr : qpos_adr+3]
            quat = state.data.qpos[qpos_adr+3 : qpos_adr+7]
            update_body_xml(active_xml_path, body_name, pos, quat)
            if state.is_light_selected:
                light_idx = get_light_idx_for_body(pm.active_body_id)
                if light_idx >= 0:
                    rgb = state.model.light_diffuse[light_idx]
                    update_light_xml(active_xml_path, body_name, rgb)
        pm.confirm_placement(state.model) 
        
        state.selected_body_id = -1
        state.selected_qpos_adr = -1
        state.is_light_selected = False
        state.listbox_index = -1 
        state.gui.set_status("Placed & Saved.")

def update_gravity_from_gui(val):
    if state.model: state.model.opt.gravity[:] = [0, 0, -float(val)]

# --- Wrapper Actions ---
def open_scene(): state.defer(_open_scene_logic)
def import_obj_workflow(obj_path=None): state.defer(_import_obj_workflow_logic, obj_path)
def add_light_workflow(): state.defer(_add_light_workflow_logic)
def delete_selected_object(): state.defer(_delete_selected_object_logic)
def perform_undo(): state.defer(_perform_undo_logic)
def perform_redo(): state.defer(_perform_redo_logic)
def confirm_current_placement(): state.defer(_confirm_current_placement_logic)
def change_floor_workflow(): state.defer(_change_floor_workflow_logic)
def export_project_workflow(): state.defer(_export_project_logic) 

def save_scene_as(): 
    batch_update_bodies_xml(active_xml_path, state.model, state.data)
    path = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML", "*.xml")])
    if path: shutil.copy(active_xml_path, path); state.gui.set_status(f"Saved to {os.path.basename(path)}")

def update_light_color_from_gui(r, g, b):
    if state.selected_body_id == -1 or not state.is_light_selected: return
    light_idx = get_light_idx_for_body(state.selected_body_id)
    if light_idx >= 0:
        state.model.light_diffuse[light_idx] = np.array([r, g, b])
        state.model.light_specular[light_idx] = np.array([r*0.5, g*0.5, b*0.5])
        geom_adr = state.model.body_geomadr[state.selected_body_id]
        geom_num = state.model.body_geomnum[state.selected_body_id]
        for i in range(geom_num):
            state.model.geom_rgba[geom_adr+i] = np.array([r, g, b, 0.3])
        mujoco.mj_forward(state.model, state.data)

def update_transform_from_gui(z, s, r, p, y):
    if state.selected_qpos_adr < 0: return
    state.current_z_height = z
    state.data.qpos[state.selected_qpos_adr + 2] = z
    state.data.qpos[state.selected_qpos_adr+3 : state.selected_qpos_adr+7] = euler2quat(r, p, y)
    scale_mgr.apply_scale(state.model, state.ctx, state.selected_body_id, s)
    mujoco.mj_forward(state.model, state.data)

def select_object_by_id(body_id):
    if pm.active_body_id != -1 and pm.active_body_id != body_id:
        cancel_active_object() 

    state.selected_body_id = body_id
    jntadr = state.model.body_jntadr[body_id]
    state.is_light_selected = (get_light_idx_for_body(body_id) != -1)
    
    if jntadr >= 0:
        qpos_adr = state.model.jnt_qposadr[jntadr]
        state.selected_qpos_adr = qpos_adr 
        
        z = state.data.qpos[qpos_adr + 2]
        quat = state.data.qpos[qpos_adr + 3 : qpos_adr + 7]
        s = scale_mgr.get_current_scale(state.model, body_id)
        r, p, y = quat2euler(quat)
        
        state.current_z_height = z
        state.current_scale = s
        state.current_roll = r
        state.current_pitch = p
        state.current_yaw = y
        
        if state.is_light_selected:
            light_idx = get_light_idx_for_body(body_id)
            if light_idx >= 0:
                rgb = state.model.light_diffuse[light_idx]
                state.current_rgb = (rgb[0], rgb[1], rgb[2])

        if body_id in state.listbox_body_ids:
            state.listbox_index = state.listbox_body_ids.index(body_id)
        else:
            state.listbox_index = -1
            
        pm.start_placement(state.model, state.data, body_id)
    else: state.selected_qpos_adr = -1

# --- Interaction (Callbacks) ---
def pick_object(window, xpos, ypos):
    width, height = glfw.get_framebuffer_size(window)
    selpnt = np.zeros(3); selgeom = np.zeros(1, dtype=np.int32); selflex = np.zeros(1, dtype=np.int32); selskin = np.zeros(1, dtype=np.int32)
    mujoco.mjv_select(state.model, state.data, state.opt, width/height, xpos/width, (height-ypos)/height, state.scn, selpnt, selgeom, selflex, selskin)
    
    geom_id = selgeom[0]
    if geom_id < 0:
        if pm.active_body_id != -1: cancel_active_object()
        return
    
    if state.model.geom_group[geom_id] == 4: return
    body_id = state.model.geom_bodyid[geom_id]
    if body_id > 0:
        select_object_by_id(body_id)
        if state.gui: state.gui.set_status(f"Selected: {body_id}")

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
    state.button_right_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS)
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
            
    elif state.button_right_pressed:
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

def recover_corrupted_scene():
    print("\n[System] ⚠️ CRITICAL: Scene corrupted. Resetting to Default...")
    if state.gui: state.gui.set_status("Scene Corrupted! Resetting...")
    initialize_project()
    load_model(restore=False)

# --- Main ---
def main():
    os.environ["MUJOCO_GL"] = "glfw"
    setup_logging()
    if not glfw.init(): return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(1200, 900, "Final Project Editor (Stable Loop)", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    state.window = window
    glfw.swap_interval(1)

    gui_callbacks = {
        'load': import_obj_workflow, 'open': open_scene, 'save': save_scene_as,
        'export': export_project_workflow, 'floor': change_floor_workflow,
        'add_light': add_light_workflow, 'undo': perform_undo, 'redo': perform_redo,
        'confirm': confirm_current_placement, 'delete': delete_selected_object,
        'list_select': on_gui_list_select, 'gravity': update_gravity_from_gui,
        'transform': update_transform_from_gui, 'light_color': update_light_color_from_gui
    }

    try:
        load_model(restore=False)
        if state.data is None: recover_corrupted_scene()
    except Exception as e:
        print(f"Initial Load Error: {e}")
        recover_corrupted_scene()

    print("[Main] Initializing ImGui...")
    state.gui = ImGuiPanel(window)

    state.cam.azimuth = 90; state.cam.elevation = -45; state.cam.distance = 10
    state.cam.lookat = np.array([0.0, 0.0, 0.0])

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    print("[Main] Entering Loop...")
    frame_count = 0

    while not glfw.window_should_close(window):
        #  動態 FPS 同步機制
        current_fps = state.target_fps if state.target_fps > 5 else 5.0
        target_dt = 1.0 / current_fps
        loop_start_time = time.time()

        glfw.poll_events()
        
        while state.pending_tasks:
            task = state.pending_tasks.pop(0)
            try: task()
            except Exception as e: print(f"Task Error: {e}")

        state.gui.process_inputs()

        try:
            #  移除 paused 檢查，物理永遠執行
            sim_start = state.data.time
            max_steps = 10; steps = 0
            while (state.data.time - sim_start < target_dt) and (steps < max_steps):
                pm.update(state.model, state.data)
                state.data.qfrc_applied[:] = 0 
                
                # 處理選取中的物體 (懸浮/無重力)
                if pm.active_body_id != -1:
                    for i in range(state.model.nbody):
                        if i == 0: continue
                        jnt_adr = state.model.body_jntadr[i]
                        if jnt_adr >= 0 and state.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                            dof_adr = state.model.jnt_dofadr[jnt_adr]
                            state.data.qvel[dof_adr : dof_adr+6] = 0
                            state.data.qfrc_applied[dof_adr : dof_adr+6] = state.data.qfrc_bias[dof_adr : dof_adr+6]
                
                # 燈光靜止邏輯
                for i in range(state.model.nlight):
                    body_id = state.model.light_bodyid[i]
                    jnt_adr = state.model.body_jntadr[body_id]
                    if jnt_adr >= 0:
                        dof_adr = state.model.jnt_dofadr[jnt_adr]
                        state.data.qvel[dof_adr : dof_adr+6] = 0
                        state.data.qfrc_applied[dof_adr : dof_adr+6] = state.data.qfrc_bias[dof_adr : dof_adr+6]
                
                # 邊界重置邏輯
                for i in range(state.model.nbody):
                    jnt_adr = state.model.body_jntadr[i]
                    if jnt_adr >= 0 and state.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                        qpos_adr = state.model.jnt_qposadr[jnt_adr]
                        if state.data.qpos[qpos_adr+2] < -5.0:
                            state.data.qpos[qpos_adr:qpos_adr+3] = [0,0,5]
                            state.data.qvel[:] = 0
                
                mujoco.mj_step(state.model, state.data)
                steps += 1
            
            state.is_placing = (pm.active_body_id != -1)
            state.is_valid = pm.is_valid
            if time.time() - state.last_auto_save_time > 1.0:
                batch_update_bodies_xml(active_xml_path, state.model, state.data)
                state.last_auto_save_time = time.time()
        except Exception as e:
            if frame_count % 60 == 0: print(f"[Physics Warning] {e}")

        try:
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(state.model, state.data, state.opt, None, state.cam, mujoco.mjtCatBit.mjCAT_ALL.value, state.scn)
            if state.selected_body_id != -1:
                for i in range(state.scn.ngeom):
                    g = state.scn.geoms[i]
                    if g.objid == -1: continue
                    bid = state.model.geom_bodyid[g.objid]
                    if bid == state.selected_body_id: g.emission += 0.3
            mujoco.mjr_render(viewport, state.scn, state.ctx)
            state.gui.render(state, gui_callbacks)
            glfw.swap_buffers(window)
        except Exception as e:
            print(f"[Render Error] {e}")
            time.sleep(0.1)

        #  FPS 鎖定
        elapsed = time.time() - loop_start_time
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

        frame_count += 1

    if state.gui: state.gui.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
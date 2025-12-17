# src/main_final.py
import mujoco
import glfw
import numpy as np
import os
import sys
import math
import shutil
from tkinter import filedialog

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loader import load_scene_with_object, delete_body_from_scene, update_body_xml, LAST_IMPORTED_BODY_NAME
from src.importer import convert_obj_with_obj2mjcf
from src.utils import euler2quat, quat2euler, save_simulation_state, restore_simulation_state
from src.managers import ScaleManager, PlacementManager, HistoryManager
from src.gui import ControlPanel

# --- Setup ---
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

BASE_XML_PATH = "scene/main_scene.xml"
CURRENT_SCENE_XML = "scene/current_scene.xml"
GRID_SIZE = 0.5

if not os.path.exists("scene"): os.makedirs("scene")
if os.path.exists(BASE_XML_PATH):
    shutil.copy(BASE_XML_PATH, CURRENT_SCENE_XML)

active_xml_path = CURRENT_SCENE_XML

# --- Global State ---
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

# --- Managers Instantiation ---
scale_mgr = ScaleManager()
pm = PlacementManager()

# --- Helper Functions ---
def load_model(restore=True):
    print(f"[System] Reloading model... (Restore={restore})")
    old_state = save_simulation_state(state.model, state.data) if restore else None
    
    try:
        state.model = mujoco.MjModel.from_xml_path(active_xml_path)
        state.data = mujoco.MjData(state.model)
        
        if restore and old_state:
            restore_simulation_state(state.model, state.data, old_state)
            
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        state.selected_body_id = -1
        state.selected_qpos_adr = -1
        
        return state.model, state.data
    except Exception as e: 
        print(f"Error loading: {e}")
        return None, None

# 初始化 HistoryManager (將 load_model 傳進去)
history = HistoryManager(reload_callback=load_model)

# --- Actions ---
def open_scene():
    path = filedialog.askopenfilename(filetypes=[("XML", "*.xml")])
    if not path: return
    shutil.copy(path, CURRENT_SCENE_XML)
    history.undo_stack.clear(); history.redo_stack.clear()
    pm.active_body_id = -1; state.selected_body_id = -1
    load_model(restore=False)
    if state.gui: state.gui.set_status(f"Opened: {os.path.basename(path)}")

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
        
        state.model = load_scene_with_object(active_xml_path, str(mjcf_path), spawn_height=spawn_h, save_merged_xml=active_xml_path)
        state.data = mujoco.MjData(state.model)
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 這裡不需要手動 restore，因為 load_scene_with_object 已經合併了 XML
        
        if LAST_IMPORTED_BODY_NAME:
            bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, LAST_IMPORTED_BODY_NAME)
            if bid >= 0:
                select_object_by_id(bid)
                if state.gui: state.gui.set_status(f"Imported: {os.path.basename(obj_path)}")
    except Exception as e:
        print(e); import traceback; traceback.print_exc()

def delete_selected_object():
    if state.selected_body_id == -1: return
    body_name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, state.selected_body_id)
    if not body_name: return
    
    history.push_state(active_xml_path, state.model, state.data)
    
    if delete_body_from_scene(active_xml_path, body_name):
        load_model(restore=True)
        pm.active_body_id = -1
        state.selected_body_id = -1
        if state.gui: state.gui.set_status(f"Deleted: {body_name}")

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

def save_scene_as():
    path = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML", "*.xml")])
    if path:
        shutil.copy(active_xml_path, path)
        if state.gui: state.gui.set_status(f"Saved to {os.path.basename(path)}")

def perform_undo():
    history.undo(active_xml_path, state.model, state.data)
    # Undo 後 context 已經在 HistoryManager 透過 load_model 更新了
    # 但我們需要重建 scene/ctx 因為 load_model 回傳了新的 model
    # 其實 load_model 已經做了，我們只需要重置選擇
    pm.active_body_id = -1

def perform_redo():
    history.redo(active_xml_path, state.model, state.data)
    pm.active_body_id = -1

# --- Interaction ---
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

# --- Callbacks ---
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

# --- Main ---
def main():
    if not glfw.init(): return
    window = glfw.create_window(1200, 900, "Final Project Editor", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    state.window = window

    state.gui = ControlPanel(import_obj_workflow, open_scene, update_transform_from_gui, confirm_current_placement, delete_selected_object, save_scene_as, perform_undo, perform_redo)

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
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

# Import loader functions
from src.loader import (
    load_scene_with_object, delete_body_from_scene, update_body_xml, 
    add_light_to_scene, update_light_xml, LAST_IMPORTED_BODY_NAME
)
from src.importer import convert_obj_with_obj2mjcf
from src.utils import euler2quat, quat2euler, save_simulation_state, restore_simulation_state
from src.managers import ScaleManager, PlacementManager, HistoryManager
from src.gui import ControlPanel
# [新增] 引入初始化模組
from src.initializer import initialize_project

# --- Setup ---
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
        self.is_dragging = False; self.button_left_pressed = False; self.shift_pressed = False
        self.window = None; self.model = None; self.data = None
        self.cam = mujoco.MjvCamera(); self.opt = mujoco.MjvOption()
        self.scn = None; self.ctx = None
        self.selected_body_id = -1; self.selected_qpos_adr = -1
        self.gui = None
        self.current_z_height = 0.0
        self.is_light_selected = False
        self.listbox_body_ids = [] # [New] 列表索引對應的 body id

state = EditorState()

# --- Managers Instantiation ---
scale_mgr = ScaleManager()
pm = PlacementManager()

# --- Helper Functions ---
def refresh_object_list_ui():
    """刷新 GUI 的物件列表 (過濾掉 World 和 Light)"""
    if not state.model or not state.gui: return
    
    names = []
    state.listbox_body_ids = []
    
    for i in range(state.model.nbody):
        # 跳過 world (id 0)
        if i == 0: continue
        
        # 跳過燈光 (檢查 light_bodyid 列表)
        is_light = False
        for l in range(state.model.nlight):
            if state.model.light_bodyid[l] == i:
                is_light = True
                break
        if is_light: continue
        
        # 取得名稱
        name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name: name = f"Body {i}"
        
        names.append(name)
        state.listbox_body_ids.append(i)
        
    state.gui.update_object_list(names)

def on_gui_list_select(index):
    """當點選列表時觸發"""
    if 0 <= index < len(state.listbox_body_ids):
        body_id = state.listbox_body_ids[index]
        select_object_by_id(body_id)

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
        state.is_light_selected = False
        
        # [New] 載入後刷新列表
        refresh_object_list_ui()
        
        return state.model, state.data
    except Exception as e: 
        print(f"Error loading: {e}")
        return None, None

history = HistoryManager(reload_callback=load_model)

# --- Action Helpers ---
def get_light_idx_for_body(body_id):
    if state.model is None: return -1
    for i in range(state.model.nlight):
        if state.model.light_bodyid[i] == body_id:
            return i
    return -1

# [新增] 封裝一個通用的「結束當前選取」邏輯
def finalize_active_object():
    """嘗試結束當前物體的編輯。回傳 True 表示成功結束，False 表示異常。"""
    if pm.active_body_id == -1: return True

    if pm.is_valid:
        # 1. 合法 -> 確認並存檔
        confirm_current_placement()
    else:
        # 2. 非法 -> 放棄修改，彈回原點
        print(f"[Auto-Revert] Body {pm.active_body_id} overlap. Reverting...")
        pm.revert_placement(state.model, state.data)
        if state.gui: state.gui.set_status("⚠️ Overlap detected! Reverted.")
        
        # 清除選取狀態
        state.selected_body_id = -1
        state.selected_qpos_adr = -1
        state.is_light_selected = False
        
    return True

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
        finalize_active_object()
        history.push_state(active_xml_path, state.model, state.data)

        mjcf_path = convert_obj_with_obj2mjcf(obj_path)
        spawn_h = state.current_z_height if state.current_z_height > 0 else 0.5
        
        state.model = load_scene_with_object(active_xml_path, str(mjcf_path), spawn_height=spawn_h, save_merged_xml=active_xml_path)
        state.data = mujoco.MjData(state.model)
        state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
        state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # [New] 匯入後刷新列表
        refresh_object_list_ui()
        
        if LAST_IMPORTED_BODY_NAME:
            bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, LAST_IMPORTED_BODY_NAME)
            if bid >= 0:
                select_object_by_id(bid)
                if state.gui: state.gui.set_status(f"Imported: {os.path.basename(obj_path)}")
    except Exception as e:
        print(e); import traceback; traceback.print_exc()

def add_light_workflow():
    try:
        if state.gui: state.gui.set_status("Adding Light...")
        finalize_active_object()

        history.push_state(active_xml_path, state.model, state.data)
        
        spawn_h = state.current_z_height if state.current_z_height > 0 else 3.0
        if add_light_to_scene(active_xml_path, spawn_pos=f"0 0 {spawn_h}"):
            load_model(restore=True)
            if LAST_IMPORTED_BODY_NAME:
                bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, LAST_IMPORTED_BODY_NAME)
                if bid >= 0:
                    select_object_by_id(bid)
                    if state.gui: state.gui.set_status("Added Point Light")
    except Exception as e:
        print(e)

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
            
            if state.is_light_selected:
                light_idx = get_light_idx_for_body(pm.active_body_id)
                if light_idx >= 0:
                    rgb = state.model.light_diffuse[light_idx]
                    update_light_xml(active_xml_path, body_name, rgb)
        
        pm.confirm_placement(state.model)
        state.selected_body_id = -1
        state.selected_qpos_adr = -1
        state.is_light_selected = False
        state.gui.set_status("Placed & Saved.")

def save_scene_as():
    path = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML", "*.xml")])
    if path:
        shutil.copy(active_xml_path, path)
        if state.gui: state.gui.set_status(f"Saved to {os.path.basename(path)}")

def perform_undo():
    history.undo(active_xml_path, state.model, state.data)
    state.scn = mujoco.MjvScene(state.model, maxgeom=10000)
    state.ctx = mujoco.MjrContext(state.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    pm.active_body_id = -1
    # [New] Undo 後刷新列表
    refresh_object_list_ui()

def perform_redo():
    history.redo(active_xml_path, state.model, state.data)
    pm.active_body_id = -1
    # [New] Redo 後刷新列表
    refresh_object_list_ui()

# --- Interaction ---
def select_object_by_id(body_id):
    if pm.active_body_id != -1 and pm.active_body_id != body_id:
        finalize_active_object()

    state.selected_body_id = body_id
    jntadr = state.model.body_jntadr[body_id]
    
    state.is_light_selected = (get_light_idx_for_body(body_id) != -1)
    
    if jntadr >= 0:
        state.selected_qpos_adr = state.model.jnt_qposadr[jntadr]
        z = state.data.qpos[state.selected_qpos_adr + 2]
        quat = state.data.qpos[state.selected_qpos_adr+3 : state.selected_qpos_adr+7]
        r, p, y = quat2euler(quat)
        s = scale_mgr.get_current_scale(state.model, body_id)
        
        state.current_z_height = z 
        
        if state.gui: state.gui.set_values(z, s, r, p, y)
        if state.is_light_selected and state.gui:
            light_idx = get_light_idx_for_body(body_id)
            rgb = state.model.light_diffuse[light_idx]
            state.gui.set_light_values(rgb[0], rgb[1], rgb[2])
            
        # [New] 同步列表選擇狀態
        if body_id in state.listbox_body_ids:
            idx = state.listbox_body_ids.index(body_id)
            if state.gui: state.gui.select_list_item(idx)
        else:
            if state.gui: state.gui.select_list_item(-1)
            
        # [Updated] 這裡傳入 state.data 讓 PM 記錄初始位置
        pm.start_placement(state.model, state.data, body_id)
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

def main():
    if not glfw.init(): return
    window = glfw.create_window(1200, 900, "Final Project Editor", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    state.window = window

    # 傳入新增的 callback
    state.gui = ControlPanel(
        load_cb=import_obj_workflow, 
        open_cb=open_scene, 
        add_light_cb=add_light_workflow,
        rot_cb=update_transform_from_gui, 
        light_color_cb=update_light_color_from_gui,
        confirm_cb=confirm_current_placement, 
        delete_cb=delete_selected_object, 
        save_cb=save_scene_as, 
        undo_cb=perform_undo, 
        redo_cb=perform_redo,
        list_select_cb=on_gui_list_select # [New]
    )

    # ==== 必須手動載入一次空場景 ====
    if os.path.exists(active_xml_path):
        print("[Main] Loading initial scene...")
        load_model(restore=False)
    else:
        print("[Main] Error: Initial scene not found!")

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
            state.gui.update_gui_state(pm.active_body_id != -1, pm.is_valid, state.selected_body_id != -1, state.is_light_selected)

            state.data.qfrc_applied[:] = 0 
            
            # 1. 放置模式：凍結所有物體
            if pm.active_body_id != -1:
                for i in range(state.model.nbody):
                    if i == 0: continue
                    jnt_adr = state.model.body_jntadr[i]
                    if jnt_adr >= 0 and state.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                        dof_adr = state.model.jnt_dofadr[jnt_adr]
                        state.data.qvel[dof_adr : dof_adr+6] = 0
                        state.data.qfrc_applied[dof_adr : dof_adr+6] = state.data.qfrc_bias[dof_adr : dof_adr+6]
            
            # 2. 永遠凍結燈光
            for i in range(state.model.nlight):
                body_id = state.model.light_bodyid[i]
                jnt_adr = state.model.body_jntadr[body_id]
                if jnt_adr >= 0 and state.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                    dof_adr = state.model.jnt_dofadr[jnt_adr]
                    state.data.qvel[dof_adr : dof_adr+6] = 0
                    state.data.qfrc_applied[dof_adr : dof_adr+6] = state.data.qfrc_bias[dof_adr : dof_adr+6]

            # 3. 虛空救援
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
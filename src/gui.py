# src/gui.py
import imgui
from imgui.integrations.glfw import GlfwRenderer

class ImGuiPanel:
    def __init__(self, window):
        # 初始化 ImGui 上下文
        imgui.create_context()
        self.impl = GlfwRenderer(window)
        
        # 這些是用來暫存輸入框數值的 (ImGui 需要保持狀態)
        self.temp_gravity = 9.81

    def process_inputs(self):
        """處理滑鼠鍵盤輸入 (在每一幀呼叫)"""
        self.impl.process_inputs()

    def render(self, state, callbacks):
        """
        繪製 GUI 的主要函式
        state: 包含像 state.current_z_height, state.selected_body_id 等變數
        callbacks: 一個字典，包含 'load', 'save', 'undo' 等函式
        """
        imgui.new_frame()

        # 設定視窗位置與大小 (可選)
        imgui.set_next_window_position(10, 10, condition=imgui.ON_FIRST_USE_EVER)
        imgui.set_next_window_size(300, 700, condition=imgui.ON_FIRST_USE_EVER)

        # 開始繪製 "Control Panel" 視窗
        imgui.begin("Control Panel")

        # --- 1. File & Project ---
        if imgui.collapsing_header("File Operations", visible=True)[0]:
            if imgui.button("Open Scene"): callbacks['open']()
            imgui.same_line()
            if imgui.button("Save As"): callbacks['save']()
            
            if imgui.button("Change Floor"): callbacks['floor']()
            if imgui.button("Export Project (Zip)"): callbacks['export']()

        # --- 2. Add Objects ---
        imgui.separator()
        imgui.text("Add Objects")
        if imgui.button("Add OBJ (I)"): callbacks['load']()
        imgui.same_line()
        if imgui.button("Add Point Light"): callbacks['add_light']()

        # --- 3. Scene List (Listbox) ---
        imgui.separator()
        imgui.text("Scene Objects")
        
        # 製作 Listbox
        # 我們需要從 state 取得名稱列表
        if hasattr(state, 'object_names') and state.object_names:
            current_item = state.listbox_index if hasattr(state, 'listbox_index') else -1
            
            # ImGui 的 Listbox 邏輯
            with imgui.begin_list_box("##Objects", 280, 100) as list_box:
                for i, name in enumerate(state.object_names):
                    is_selected = (i == current_item)
                    clicked, selected = imgui.selectable(name, is_selected)
                    
                    if clicked:
                        callbacks['list_select'](i)
                    
                    # 確保選取項目可見
                    if is_selected:
                        imgui.set_item_default_focus()
        else:
            imgui.text("(No objects)")

        # --- 4. History ---
        imgui.separator()
        if imgui.button("Undo"): callbacks['undo']()
        imgui.same_line()
        if imgui.button("Redo"): callbacks['redo']()

        # --- 5. Placement & Edit ---
        imgui.separator()
        imgui.text("Placement")
        
        # 狀態顯示 (Valid/Overlap)
        if state.is_placing:
            if state.is_valid:
                imgui.text_colored("VALID POSITION", 0.2, 1.0, 0.2)
                if imgui.button("Confirm (Enter)"): callbacks['confirm']()
            else:
                imgui.text_colored("OVERLAP DETECTED", 1.0, 0.2, 0.2)
                # 碰撞時禁用 Confirm 按鈕
                imgui.begin_disabled()
                imgui.button("Confirm (Enter)")
                imgui.end_disabled()
        else:
            imgui.text_colored("Ready", 0.7, 0.7, 0.7)
            # 沒有在放置時禁用 Confirm
            imgui.begin_disabled()
            imgui.button("Confirm (Enter)")
            imgui.end_disabled()

        if state.selected_body_id != -1:
             if imgui.button("Delete (Del)"): callbacks['delete']()

        # --- 6. Physics (Gravity) ---
        imgui.separator()
        imgui.text("Physics")
        # InputFloat 回傳 (changed, value)
        changed, self.temp_gravity = imgui.input_float("Gravity Z", self.temp_gravity)
        if changed:
             callbacks['gravity'](self.temp_gravity)

        # --- 7. Transform (只有選取物體時顯示) ---
        if state.selected_body_id != -1:
            imgui.separator()
            imgui.text(f"Transform (Body {state.selected_body_id})")

            # Height (Z)
            changed, new_z = imgui.slider_float("Height", state.current_z_height, 0.0, 5.0)
            if changed:
                # 這裡要呼叫更新 transform 的 callback，但因為我們有多個數值
                # 最簡單的方式是傳遞變更後的數值
                callbacks['transform'](new_z, state.current_scale, state.current_roll, state.current_pitch, state.current_yaw)

            # Scale
            changed, new_scale = imgui.slider_float("Scale", state.current_scale, 0.1, 3.0)
            if changed:
                callbacks['transform'](state.current_z_height, new_scale, state.current_roll, state.current_pitch, state.current_yaw)
            
            # Rotations
            c_r, new_roll = imgui.slider_float("Roll (X)", state.current_roll, -180, 180)
            c_p, new_pitch = imgui.slider_float("Pitch (Y)", state.current_pitch, -180, 180)
            c_y, new_yaw = imgui.slider_float("Yaw (Z)", state.current_yaw, -180, 180)
            
            if c_r or c_p or c_y:
                callbacks['transform'](state.current_z_height, state.current_scale, new_roll, new_pitch, new_yaw)

            # --- 8. Light Color (只有選取燈光時顯示) ---
            if state.is_light_selected:
                imgui.separator()
                imgui.text("Light Color")
                # ColorEdit3 回傳 (changed, (r, g, b))
                # 假設 state 有 current_rgb = (1.0, 1.0, 1.0)
                if not hasattr(state, 'current_rgb'): state.current_rgb = (1.0, 1.0, 1.0)
                
                changed, new_color = imgui.color_edit3("Diffuse", *state.current_rgb)
                if changed:
                    state.current_rgb = new_color # 立即更新顯示
                    callbacks['light_color'](*new_color)

        imgui.end() # End "Control Panel"

        # 結束並提交繪製指令
        imgui.render()
        self.impl.render(imgui.get_draw_data())
    
    def shutdown(self):
        self.impl.shutdown()
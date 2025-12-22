# src/gui.py
import imgui
from imgui.integrations.glfw import GlfwRenderer
import sys

class ImGuiPanel:
    def __init__(self, window):
        imgui.create_context()
        self.impl = GlfwRenderer(window)
        self.temp_gravity = 9.81
        self.status_message = ""

    def set_status(self, message):
        self.status_message = message

    def process_inputs(self):
        self.impl.process_inputs()

    def render(self, state, callbacks):
        try:
            imgui.new_frame()

            imgui.set_next_window_position(10, 10, condition=imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(300, 700, condition=imgui.FIRST_USE_EVER)

            imgui.begin("Control Panel")

            # --- Status Bar ---
            if self.status_message:
                imgui.text_colored(self.status_message, 1.0, 1.0, 0.0)
                imgui.separator()

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

            # --- 3. Scene List ---
            imgui.separator()
            imgui.text("Scene Objects")
            if hasattr(state, 'object_names') and state.object_names:
                current_item = state.listbox_index if hasattr(state, 'listbox_index') else -1
                
                with imgui.begin_list_box("##Objects", 280, 150) as list_box:
                    for i, name in enumerate(state.object_names):
                        is_selected = (i == current_item)
                        if imgui.selectable(name, is_selected)[1]: 
                            callbacks['list_select'](i)
                        if is_selected:
                            imgui.set_item_default_focus()
            else:
                imgui.text("(No objects)")

            # --- 4. History ---
            imgui.separator()
            if imgui.button("Undo"): callbacks['undo']()
            imgui.same_line()
            if imgui.button("Redo"): callbacks['redo']()

            # --- 5. Placement ---
            imgui.separator()
            imgui.text("Placement")
            
            def safe_disabled(disabled, func):
                if disabled:
                    if hasattr(imgui, 'begin_disabled'): imgui.begin_disabled()
                    else: imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                func()
                if disabled:
                    if hasattr(imgui, 'end_disabled'): imgui.end_disabled()
                    else: imgui.pop_style_var()

            if state.is_placing:
                if state.is_valid:
                    imgui.text_colored("VALID POSITION", 0.2, 1.0, 0.2)
                    if imgui.button("Confirm (Enter)"): callbacks['confirm']()
                else:
                    imgui.text_colored("OVERLAP DETECTED", 1.0, 0.2, 0.2)
                    safe_disabled(True, lambda: imgui.button("Confirm (Enter)"))
            else:
                imgui.text_colored("Ready", 0.7, 0.7, 0.7)
                safe_disabled(True, lambda: imgui.button("Confirm (Enter)"))

            if state.selected_body_id != -1:
                 if imgui.button("Delete (Del)"): callbacks['delete']()

            # --- 6. Physics (Modified: Gravity Button) ---
            imgui.separator()
            imgui.text("Physics")
            # [修正] 這裡只更新暫存變數，不直接觸發 callback
            _, self.temp_gravity = imgui.input_float("Gravity Z", self.temp_gravity)
            imgui.same_line()
            # [修正] 新增 Set 按鈕，按下才觸發更新
            if imgui.button("Set##Grav"): 
                callbacks['gravity'](self.temp_gravity)

            # --- 7. Transform ---
            if state.selected_body_id != -1:
                imgui.separator()
                imgui.text(f"Transform (Body {state.selected_body_id})")

                z_val = state.current_z_height if hasattr(state, 'current_z_height') else 0.0
                scale_val = state.current_scale if hasattr(state, 'current_scale') else 1.0
                roll_val = state.current_roll if hasattr(state, 'current_roll') else 0.0
                pitch_val = state.current_pitch if hasattr(state, 'current_pitch') else 0.0
                yaw_val = state.current_yaw if hasattr(state, 'current_yaw') else 0.0

                changed, new_z = imgui.slider_float("Height", z_val, 0.0, 5.0)
                if changed: callbacks['transform'](new_z, scale_val, roll_val, pitch_val, yaw_val)

                changed, new_scale = imgui.slider_float("Scale", scale_val, 0.1, 3.0)
                if changed: callbacks['transform'](z_val, new_scale, roll_val, pitch_val, yaw_val)
                
                c_r, new_roll = imgui.slider_float("Roll (X)", roll_val, -180, 180)
                c_p, new_pitch = imgui.slider_float("Pitch (Y)", pitch_val, -180, 180)
                c_y, new_yaw = imgui.slider_float("Yaw (Z)", yaw_val, -180, 180)
                if c_r or c_p or c_y:
                    callbacks['transform'](z_val, scale_val, new_roll, new_pitch, new_yaw)

                # --- 8. Light Color ---
                if state.is_light_selected:
                    imgui.separator()
                    imgui.text("Light Color")
                    if not hasattr(state, 'current_rgb'): state.current_rgb = (1.0, 1.0, 1.0)
                    changed, new_color = imgui.color_edit3("Diffuse", *state.current_rgb)
                    if changed:
                        state.current_rgb = new_color
                        callbacks['light_color'](*new_color)

            imgui.end()
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            
        except Exception as e:
            print(f"[GUI Error] {e}")
            try: imgui.end_frame()
            except: pass
    
    def shutdown(self):
        self.impl.shutdown()
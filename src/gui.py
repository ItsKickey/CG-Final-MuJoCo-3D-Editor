# src/gui.py
import tkinter as tk

class ControlPanel:
    def __init__(self, load_cb, open_cb, add_light_cb, rot_cb, light_color_cb, confirm_cb, delete_cb, save_cb, undo_cb, redo_cb):
        self.root = tk.Tk()
        self.root.title("Control Panel")
        self.root.geometry("360x800") # 寬度稍微加寬，高度設為固定
        self.root.configure(bg="#f0f0f0")
        self.root.attributes("-topmost", True)

        # ==== [新增] 滾動條容器設置 ====
        # 1. 建立外層 Frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True)

        # 2. 建立 Canvas (用於滾動)
        self.canvas = tk.Canvas(main_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        # 3. 建立 Scrollbar
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # 4. 綁定 Canvas 與 Scrollbar
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # 5. 建立實際內容的 Frame (所有按鈕都放這裡)
        self.content_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        
        # 將內容 Frame 放入 Canvas 視窗中
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # 6. 事件綁定：更新滾動區域
        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        def on_canvas_configure(event):
            # 讓內容寬度跟隨視窗寬度調整
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        self.content_frame.bind("<Configure>", on_frame_configure)
        self.canvas.bind("<Configure>", on_canvas_configure)

        # 7. 滑鼠滾輪綁定
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # 全域綁定滾輪 (只要滑鼠在程式上就能滾動)
        self.root.bind_all("<MouseWheel>", _on_mousewheel)

        # ==== 介面內容 (注意 parent 改成 self.content_frame) ====

        tk.Label(self.content_frame, text="Furniture Placer", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        # File
        frame_file = tk.LabelFrame(self.content_frame, text="File", padx=10, pady=10, bg="#f0f0f0")
        frame_file.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_file, text="📂 Open Scene...", command=open_cb, bg="#e1e1e1").pack(fill="x", pady=2)
        tk.Button(frame_file, text="💾 Save Scene As...", command=save_cb, bg="#ccf").pack(fill="x", pady=2)
        
        # Add Object
        frame_add = tk.LabelFrame(self.content_frame, text="Add", padx=10, pady=10, bg="#f0f0f0")
        frame_add.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_add, text="➕ Add OBJ (I)", command=load_cb, bg="#e1e1e1", height=2).pack(fill="x", pady=2)
        tk.Button(frame_add, text="💡 Add Point Light", command=add_light_cb, bg="#ffecb3").pack(fill="x", pady=2)

        # History
        frame_hist = tk.LabelFrame(self.content_frame, text="History", padx=10, pady=10, bg="#f0f0f0")
        frame_hist.pack(fill="x", padx=10, pady=5)
        btn_undo = tk.Button(frame_hist, text="↩ Undo (Ctrl+Z)", command=undo_cb, width=15)
        btn_undo.pack(side="left", padx=5)
        btn_redo = tk.Button(frame_hist, text="↪ Redo (Ctrl+Y)", command=redo_cb, width=15)
        btn_redo.pack(side="right", padx=5)

        # Placement
        self.frame_place = tk.LabelFrame(self.content_frame, text="Placement", padx=10, pady=10, bg="#f0f0f0")
        self.frame_place.pack(fill="x", padx=10, pady=5)
        self.lbl_validity = tk.Label(self.frame_place, text="Waiting...", font=("Arial", 12, "bold"), bg="#ddd", width=15)
        self.lbl_validity.pack(pady=5)
        self.btn_confirm = tk.Button(self.frame_place, text="✅ Confirm (Enter)", command=confirm_cb, bg="#8f8", state="disabled", height=2)
        self.btn_confirm.pack(fill="x", pady=5)
        self.btn_delete = tk.Button(self.frame_place, text="🗑️ Delete (Del)", command=delete_cb, bg="#f88", state="disabled", height=2)
        self.btn_delete.pack(fill="x", pady=5)

        # Transform
        frame_trans = tk.LabelFrame(self.content_frame, text="Transform", padx=10, pady=10, bg="#f0f0f0")
        frame_trans.pack(fill="x", padx=10, pady=5)
        self.var_z = tk.DoubleVar(); self.var_scale = tk.DoubleVar(value=1.0)
        self.var_roll = tk.DoubleVar(); self.var_pitch = tk.DoubleVar(); self.var_yaw = tk.DoubleVar()
        self.is_updating = False

        def on_trans_change(_):
            if not self.is_updating:
                rot_cb(self.var_z.get(), self.var_scale.get(), self.var_roll.get(), self.var_pitch.get(), self.var_yaw.get())

        tk.Label(frame_trans, text="Height (Z)", bg="#f0f0f0", fg="blue").pack(anchor="w")
        self.s_z = tk.Scale(frame_trans, variable=self.var_z, from_=0.0, to=3.0, resolution=0.05, orient="horizontal", command=on_trans_change, bg="#f0f0f0")
        self.s_z.pack(fill="x")
        tk.Label(frame_trans, text="Scale", bg="#f0f0f0", fg="red").pack(anchor="w")
        self.s_scale = tk.Scale(frame_trans, variable=self.var_scale, from_=0.1, to=3.0, resolution=0.1, orient="horizontal", command=on_trans_change, bg="#f0f0f0")
        self.s_scale.pack(fill="x")
        for label, var in [("Roll (X)", self.var_roll), ("Pitch (Y)", self.var_pitch), ("Yaw (Z)", self.var_yaw)]:
            tk.Label(frame_trans, text=label, bg="#f0f0f0").pack(anchor="w")
            tk.Scale(frame_trans, variable=var, from_=-180, to=180, orient="horizontal", command=on_trans_change, bg="#f0f0f0").pack(fill="x")

        # Light Control
        self.frame_light = tk.LabelFrame(self.content_frame, text="Light Color", padx=10, pady=10, bg="#f0f0f0")
        self.frame_light.pack(fill="x", padx=10, pady=5)
        
        self.var_lr = tk.DoubleVar(value=1.0)
        self.var_lg = tk.DoubleVar(value=1.0)
        self.var_lb = tk.DoubleVar(value=1.0)

        def on_light_change(_):
            if not self.is_updating:
                light_color_cb(self.var_lr.get(), self.var_lg.get(), self.var_lb.get())

        for label, var in [("Red", self.var_lr), ("Green", self.var_lg), ("Blue", self.var_lb)]:
            tk.Label(self.frame_light, text=label, bg="#f0f0f0").pack(anchor="w")
            tk.Scale(self.frame_light, variable=var, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", command=on_light_change, bg="#f0f0f0").pack(fill="x")

        self.lbl_status = tk.Label(self.content_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor="w")
        self.lbl_status.pack(side="bottom", fill="x", pady=(20, 0))

    def update_gui_state(self, is_placing, is_valid, has_selection, is_light_selected):
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
        
        state_light = "normal" if is_light_selected else "disabled"
        for child in self.frame_light.winfo_children():
            try: child.configure(state=state_light)
            except: pass

    def update(self):
        try: self.root.update_idletasks(); self.root.update()
        except: pass

    def set_status(self, msg): self.lbl_status.config(text=msg)
    
    def set_values(self, z, s, r, p, y):
        self.is_updating = True
        self.var_z.set(z); self.var_scale.set(s)
        self.var_roll.set(r); self.var_pitch.set(p); self.var_yaw.set(y)
        self.is_updating = False

    def set_light_values(self, r, g, b):
        self.is_updating = True
        self.var_lr.set(r); self.var_lg.set(g); self.var_lb.set(b)
        self.is_updating = False
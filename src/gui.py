# src/gui.py
import tkinter as tk

class ControlPanel:
    def __init__(self, load_cb, open_cb, add_light_cb, rot_cb, light_color_cb, confirm_cb, delete_cb, save_cb, undo_cb, redo_cb, list_select_cb,floor_cb,gravity_cb,export_cb):
        self.root = tk.Tk()
        self.root.title("Control Panel")
        self.root.geometry("360x900")
        self.root.configure(bg="#f0f0f0")
        self.root.attributes("-topmost", True)
        
        # 保存 callback
        self.list_select_cb = list_select_cb

        # ==== 滾動條容器設置 ====
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.content_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        def on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)

        self.content_frame.bind("<Configure>", on_frame_configure)
        self.canvas.bind("<Configure>", on_canvas_configure)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.root.bind_all("<MouseWheel>", _on_mousewheel)

        # ==== 介面內容 ====

        tk.Label(self.content_frame, text="Scene Editer", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        # File
        frame_file = tk.LabelFrame(self.content_frame, text="File", padx=10, pady=10, bg="#f0f0f0")
        frame_file.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_file, text="📂 Open Scene...", command=open_cb, bg="#e1e1e1").pack(fill="x", pady=2)
        tk.Button(frame_file, text="💾 Save Scene As...", command=save_cb, bg="#ccf").pack(fill="x", pady=2)
        tk.Button(frame_file, text="🖼️ Change Floor Texture", command=floor_cb, bg="#d9d9f3").pack(fill="x", pady=2)
        tk.Button(frame_file, text="📦 Export Project (Zip)", command=export_cb, bg="#b3e5fc").pack(fill="x", pady=2)
        # Add Object
        frame_add = tk.LabelFrame(self.content_frame, text="Add", padx=10, pady=10, bg="#f0f0f0")
        frame_add.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_add, text="➕ Add OBJ (I)", command=load_cb, bg="#e1e1e1", height=2).pack(fill="x", pady=2)
        tk.Button(frame_add, text="💡 Add Point Light", command=add_light_cb, bg="#ffecb3").pack(fill="x", pady=2)

        # Scene Objects List (物件列表)
        frame_list = tk.LabelFrame(self.content_frame, text="Scene Objects", padx=10, pady=10, bg="#f0f0f0")
        frame_list.pack(fill="x", padx=10, pady=5)
        
        lb_frame = tk.Frame(frame_list, bg="#f0f0f0")
        lb_frame.pack(fill="x")
        
        self.lb_scroll = tk.Scrollbar(lb_frame, orient="vertical")
        self.lb_obj = tk.Listbox(lb_frame, height=6, exportselection=False, yscrollcommand=self.lb_scroll.set)
        self.lb_scroll.config(command=self.lb_obj.yview)
        
        self.lb_obj.pack(side="left", fill="x", expand=True)
        self.lb_scroll.pack(side="right", fill="y")
        
        # 綁定選擇事件
        self.lb_obj.bind("<<ListboxSelect>>", self.on_list_select)

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
        frame_phys = tk.LabelFrame(self.content_frame, text="Physics", padx=10, pady=10, bg="#f0f0f0")
        frame_phys.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_phys, text="Gravity Z (Input + Enter):", bg="#f0f0f0").pack(anchor="w")
        
        #Gravity 預設為 9.81
        self.var_grav = tk.DoubleVar(value=9.81) 
        
        # 建立 Entry 輸入框
        self.entry_grav = tk.Entry(frame_phys, textvariable=self.var_grav, bg="white")
        self.entry_grav.pack(fill="x", pady=2)
        
        # 定義觸發函式
        def on_gravity_enter(event=None):
            try:
                val = self.var_grav.get()
                gravity_cb(val) # 呼叫 main 傳進來的 callback
                # 為了使用者體驗，可以讓焦點離開輸入框 (Optional)
                self.content_frame.focus_set()
            except ValueError:
                pass

        # 綁定 Enter 鍵 (Return) 觸發更新
        self.entry_grav.bind("<Return>", on_gravity_enter)
        
        # 也可以加個小按鈕以免使用者不知道要按 Enter
        tk.Button(frame_phys, text="Set Gravity", command=on_gravity_enter, bg="#ddd", height=1).pack(fill="x", pady=2)

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

    def on_list_select(self, event):
        sel = self.lb_obj.curselection()
        if sel:
            index = sel[0]
            self.list_select_cb(index)

    def update_object_list(self, names):
        """更新列表內容"""
        self.lb_obj.delete(0, tk.END)
        for name in names:
            self.lb_obj.insert(tk.END, name)

    def select_list_item(self, index):
        """程式控制選擇某個項目"""
        self.lb_obj.selection_clear(0, tk.END)
        if index != -1:
            self.lb_obj.selection_set(index)
            self.lb_obj.see(index) # 確保捲動到可見

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
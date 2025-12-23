# src/initializer.py
import os
import shutil

# (保留 DEFAULT_SCENE_XML 字串內容)
DEFAULT_SCENE_XML = """
<mujoco model="My Final Project">
  <option timestep="0.002" gravity="0 0 -9.81" />

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".8 .8 .8" />
    <material name="grid_mat" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2" />
  </asset>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8" />
    
    <geom name="floor" type="plane" size="10 10 .1" material="grid_mat" />

    <body name="default_light" pos="0 0 4">
        <joint type="free" name="default_light_joint"/>
        <light mode="trackcom" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="true" dir="0 0 -1"/>
        <geom type="sphere" size="0.1" rgba="1 1 0.8 0.3" contype="0" conaffinity="0" group="1"/>
    </body>

  </worldbody>
</mujoco>
"""

def initialize_project():
    """
    初始化專案：
    每次啟動時，強制將 Default Scene 複製為 Current Scene (Runtime Cache)。
    達到類似 Blender 的 'Fresh Start' 效果。
    """
    
    default_dir = "defaultscene"
    scene_dir = "scene"
    
    base_xml = os.path.join(default_dir, "main_scene.xml")
    current_xml = os.path.join(scene_dir, "current_scene.xml") # 這就是我們的暫存檔

    # 1. 建立資料夾
    if not os.path.exists(default_dir): os.makedirs(default_dir)
    if not os.path.exists(scene_dir): os.makedirs(scene_dir)

    # 2. 確保 Default 母本存在
    if not os.path.exists(base_xml):
        print(f"[Init] Regenerating default template: {base_xml}")
        with open(base_xml, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SCENE_XML.strip())
    
    # 3. 強制重置 Current Scene
    # 不管之前有沒有 current_scene，直接用 default 覆蓋它
    print(f"[Init] Resetting runtime scene from default...")
    shutil.copy(base_xml, current_xml)
        
    return base_xml, current_xml
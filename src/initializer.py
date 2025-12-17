# src/initializer.py
import os
import shutil

# 內嵌的預設場景 XML (包含可編輯的燈光與地板)
# 這樣即使沒有任何外部檔案，程式也能產生出這個場景
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
    初始化專案結構：
    1. 檢查 defaultscene 資料夾與 main_scene.xml，沒有就自動產生。
    2. 檢查 scene 資料夾與 current_scene.xml，沒有就從 default 複製。
    """
    
    # 定義路徑
    default_dir = "defaultscene"
    scene_dir = "scene"
    
    # 這是我們的「母本」，如果 user 搞壞了，這裡永遠有一份備份
    base_xml = os.path.join(default_dir, "main_scene.xml")
    
    # 這是程式實際運行的「工作檔」
    current_xml = os.path.join(scene_dir, "current_scene.xml")

    # 1. 建立資料夾
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
        print(f"[Init] Created directory: {default_dir}")
        
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)
        print(f"[Init] Created directory: {scene_dir}")

    # 2. 檢查並重生 Base XML (防手殘機制)
    if not os.path.exists(base_xml):
        print(f"[Init] CRITICAL: {base_xml} missing! Regenerating default scene...")
        with open(base_xml, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SCENE_XML.strip())
    
    # 3. 確保 Current XML 存在 (每次啟動時，若沒有 current 則複製一份)
    # (可選：如果你希望每次重開都重置，可以在這裡強制 shutil.copy)
    if not os.path.exists(current_xml):
        print(f"[Init] Creating work scene: {current_xml}")
        shutil.copy(base_xml, current_xml)
    else:
        # 如果你想每次啟動都重置場景，請取消下面這行的註解
        # shutil.copy(base_xml, current_xml)
        pass
        
    return base_xml, current_xml
# src/importer.py
import shutil
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# [新增] 嘗試導入 obj2mjcf

try:
    from obj2mjcf.cli import process_obj, Args
except ImportError:
    # 如果執行環境的 path 設定不同，手動加入根目錄
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from obj2mjcf.cli import process_obj, Args

# 定義匯入後的統一存放路徑
if getattr(sys, 'frozen', False):
    # 如果是打包後的 EXE，根目錄應該是執行檔 (.exe) 所在的資料夾
    PROJECT_ROOT = Path(sys.executable).parent
else:
    # 如果是開發模式 (python main.py)，保持原樣
    PROJECT_ROOT = Path(__file__).parent.parent


IMPORT_DEST_DIR = PROJECT_ROOT / "Import_mjcf" 

def convert_obj_with_obj2mjcf(obj_path: str) -> Path:
    obj_path = Path(obj_path).resolve()
    obj_dir  = obj_path.parent
    obj_stem = obj_path.stem

    # 1. [修改] obj2mjcf 的 Python 函式，不再使用 subprocess
    print(f"[obj2mjcf] Processing internal: {obj_path}")
    
    # 建構參數物件，模擬 CLI 的參數
    # 對應之前的指令: --obj-dir ... --save-mjcf --overwrite
    args = Args(
        obj_dir=str(obj_dir),
        save_mjcf=True,
        overwrite=True,
    )

    try:
        # process_obj 會自動在 obj_dir 下建立一個以 obj_stem 命名的資料夾，並產生 .xml
        process_obj(obj_path, args)
    except Exception as e:
        print(f"[obj2mjcf] Error during processing: {e}")
        raise e

    # 2. 定位產生的資料夾
    # obj2mjcf 的邏輯是會在 obj 旁邊建立一個與檔名相同的資料夾
    generated_dir = obj_dir / obj_stem
    
    # 檢查是否成功建立
    if not generated_dir.exists():
        # 如果失敗，嘗試找找看有沒有散落的 XML (雖然 process_obj 通常會建資料夾)
        potential_xml = obj_dir / f"{obj_stem}.xml"
        if potential_xml.exists():
            print("[importer] Warning: obj2mjcf did not create a folder. Creating one manually.")
            generated_dir.mkdir(exist_ok=True)
            shutil.move(str(potential_xml), str(generated_dir))
        else:
            raise FileNotFoundError(f"[importer] Output directory not found: {generated_dir}")

    # 3. 搬運到專案的 Import_mjcf 目錄
    if not IMPORT_DEST_DIR.exists():
        IMPORT_DEST_DIR.mkdir(parents=True, exist_ok=True)

    target_dir = IMPORT_DEST_DIR / obj_stem
    
    if generated_dir.resolve() != target_dir.resolve():
        if target_dir.exists():
            print(f"[importer] Removing old asset at {target_dir}")
            shutil.rmtree(target_dir)
        
        print(f"[importer] Moving assets to project library: {target_dir}")
        shutil.move(str(generated_dir), str(IMPORT_DEST_DIR))
    else:
        print("[importer] Asset is already in the project library.")

    # 4. 更新 xml_path 指向新位置
    xml_path = target_dir / f"{obj_stem}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"[importer] MJCF XML not found in target: {xml_path}")

    # 5. [XML 後處理] 自動 rename (保持您原本的邏輯不變)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    prefix = obj_stem + "_"

    for tex in root.findall(".//texture"):
        old = tex.get("name")
        if old: tex.set("name", prefix + old)

    for mat in root.findall(".//material"):
        old_name = mat.get("name")
        if old_name: mat.set("name", prefix + old_name)
        old_tex = mat.get("texture")
        if old_tex: mat.set("texture", prefix + old_tex)

    for def_geom in root.findall(".//default/geom"):
        if "material" in def_geom.attrib:
            def_geom.set("material", prefix + def_geom.get("material"))

    for mesh in root.findall(".//mesh"):
        old = mesh.get("name")
        if not old:
            f = mesh.get("file")
            if f: old = Path(f).stem
            else: continue
        mesh.set("name", prefix + old)

    for geom in root.findall(".//geom"):
        if "material" in geom.attrib:
            geom.set("material", prefix + geom.get("material"))
        if "mesh" in geom.attrib:
            geom.set("mesh", prefix + geom.get("mesh"))

    tree.write(xml_path)
    print(f"[importer] Processed & Saved MJCF: {xml_path}")
    
    return xml_path
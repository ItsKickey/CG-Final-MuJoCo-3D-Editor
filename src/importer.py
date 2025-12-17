# src/importer.py
import subprocess
import shutil
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# 定義匯入後的統一存放路徑
# 假設此檔案在 src/，專案根目錄就是 src/../
PROJECT_ROOT = Path(__file__).parent.parent
IMPORT_DEST_DIR = PROJECT_ROOT / "Assets" / "imported_mjcf"

def convert_obj_with_obj2mjcf(obj_path: str) -> Path:
    obj_path = Path(obj_path).resolve()
    obj_dir  = obj_path.parent
    obj_stem = obj_path.stem

    # 1. 呼叫 obj2mjcf (在來源目錄產生，以確保能抓到相對路徑的貼圖)
    # obj2mjcf 會在 obj_dir 下產生一個名為 obj_stem 的資料夾
    cmd = [
        "obj2mjcf",
        "--obj-dir", str(obj_dir),
        "--obj-filter", obj_path.name,
        "--save-mjcf",
        "--overwrite",
    ]
    print("[obj2mjcf] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2. 定位產生的資料夾
    generated_dir = obj_dir / obj_stem
    if not generated_dir.exists():
        # 有些版本的 obj2mjcf 行為不同，如果沒產生資料夾，嘗試找 XML
        potential_xml = obj_dir / f"{obj_stem}.xml"
        if potential_xml.exists():
            # 這是個特例，如果是單檔，我們手動建資料夾包起來
            print("[importer] Warning: obj2mjcf did not create a folder. Creating one manually.")
            generated_dir.mkdir(exist_ok=True)
            shutil.move(str(potential_xml), str(generated_dir))
            # 嘗試移動相關資產 (這比較危險，可能漏搬，但暫時先這樣)
            # 建議 obj2mjcf 版本要新一點
        else:
            raise FileNotFoundError(f"[importer] Output directory not found: {generated_dir}")

    # 3. 搬運到專案的 Assets/imported_mjcf 目錄
    if not IMPORT_DEST_DIR.exists():
        IMPORT_DEST_DIR.mkdir(parents=True, exist_ok=True)

    target_dir = IMPORT_DEST_DIR / obj_stem
    
    # 防止自己搬給自己 (如果使用者已經在 imported_mjcf 裡操作)
    if generated_dir.resolve() != target_dir.resolve():
        # 如果目標已存在，先刪除舊的
        if target_dir.exists():
            print(f"[importer] Removing old asset at {target_dir}")
            shutil.rmtree(target_dir)
        
        # 搬運整包
        print(f"[importer] Moving assets to project library: {target_dir}")
        shutil.move(str(generated_dir), str(IMPORT_DEST_DIR))
    else:
        print("[importer] Asset is already in the project library.")

    # 4. 更新 xml_path 指向新位置
    xml_path = target_dir / f"{obj_stem}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"[importer] MJCF XML not found in target: {xml_path}")

    # 5. [XML 後處理] 自動 rename mesh/texture/material
    # 這一步在搬運後做，確保修改的是專案內的檔案
    tree = ET.parse(xml_path)
    root = tree.getroot()

    prefix = obj_stem + "_"   # 如 cargo_B_

    # Rename Textures
    for tex in root.findall(".//texture"):
        old = tex.get("name")
        if old: tex.set("name", prefix + old)

    # Rename Materials & Fix References
    for mat in root.findall(".//material"):
        old_name = mat.get("name")
        if old_name: mat.set("name", prefix + old_name)
        
        old_tex = mat.get("texture")
        if old_tex: mat.set("texture", prefix + old_tex)

    # Fix Material references in <default>
    for def_geom in root.findall(".//default/geom"):
        if "material" in def_geom.attrib:
            def_geom.set("material", prefix + def_geom.get("material"))

    # Rename Meshes
    for mesh in root.findall(".//mesh"):
        old = mesh.get("name")
        if not old:
            f = mesh.get("file")
            if f: old = Path(f).stem
            else: continue
        mesh.set("name", prefix + old)

    # Fix Geom References
    for geom in root.findall(".//geom"):
        if "material" in geom.attrib:
            geom.set("material", prefix + geom.get("material"))
        if "mesh" in geom.attrib:
            geom.set("mesh", prefix + geom.get("mesh"))

    tree.write(xml_path)
    print(f"[importer] Processed & Saved MJCF: {xml_path}")
    
    return xml_path
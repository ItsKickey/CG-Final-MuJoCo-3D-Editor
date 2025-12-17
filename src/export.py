# src/export.py
import os
import shutil
import zipfile
import xml.etree.ElementTree as ET

def export_project_to_zip(active_xml_path, scene_name):
    """
    將目前的場景與相關資源打包成 ZIP 檔。
    結構:
      - scene/{scene_name}.xml
      - Import_mjcf/ (僅包含用到的模型與貼圖)
    """
    print(f"[Export] Starting export for: {scene_name}")
    
    # 1. 準備路徑
    # 假設 export.py 在 src/ 下，往上兩層是專案根目錄
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputfile")
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    # 建立暫存資料夾
    temp_root = os.path.join(output_dir, "temp_export")
    if os.path.exists(temp_root): 
        shutil.rmtree(temp_root)
    os.makedirs(temp_root)

    # 建立目標結構
    target_scene_dir = os.path.join(temp_root, "scene")
    target_import_dir = os.path.join(temp_root, "Import_mjcf")
    os.makedirs(target_scene_dir)
    os.makedirs(target_import_dir)

    # 2. 解析 XML
    tree = ET.parse(active_xml_path)
    root = tree.getroot()
    active_xml_dir = os.path.dirname(os.path.abspath(active_xml_path))
    real_import_root = os.path.join(project_root, "Import_mjcf")

    copied_folders = set() # 用來記錄已複製的資料夾，避免重複複製

    def process_file_path(node, attr_name="file"):
        filename = node.get(attr_name)
        if not filename: return

        # 取得絕對路徑
        abs_source_path = os.path.abspath(os.path.join(active_xml_dir, filename))
        
        if not os.path.exists(abs_source_path):
            print(f"[Export] Warning: File not found {abs_source_path}")
            return

        # 判斷檔案是否在 Import_mjcf 內
        # 計算相對於 Import_mjcf 的路徑
        try:
            rel_in_import = os.path.relpath(abs_source_path, real_import_root)
        except ValueError:
            rel_in_import = filename # 無法計算 (可能在不同磁碟機)，視為外部檔案
        
        # 檢查是否真的在 Import_mjcf 裡面 (開頭沒有 .. 且不是絕對路徑)
        is_inside_import = (not rel_in_import.startswith("..")) and (not os.path.isabs(rel_in_import))

        if is_inside_import:
            # === Case A: 檔案在 Import_mjcf 內 ===
            
            # 特別處理: grid_textures (只複製單檔)
            path_parts = rel_in_import.split(os.sep)
            if "grid_textures" in path_parts:
                dest_path = os.path.join(target_import_dir, rel_in_import)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(abs_source_path, dest_path)
            else:
                # 一般模型: 複製整個父資料夾 (例如 cargo_A/) 以確保關聯檔案 (.mtl) 都在
                parent_dir_name = os.path.dirname(rel_in_import)
                if parent_dir_name and parent_dir_name != ".":
                    # 來源與目標資料夾路徑
                    src_folder = os.path.join(real_import_root, parent_dir_name)
                    dst_folder = os.path.join(target_import_dir, parent_dir_name)
                    
                    if src_folder not in copied_folders:
                        # 如果目標已存在先刪除 (理論上 temp 是空的，但為了保險)
                        if os.path.exists(dst_folder): shutil.rmtree(dst_folder)
                        shutil.copytree(src_folder, dst_folder)
                        copied_folders.add(src_folder)
                else:
                    # 檔案直接在 Import_mjcf 根目錄 (少見情況)，只複製檔案
                    dest_path = os.path.join(target_import_dir, rel_in_import)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(abs_source_path, dest_path)

            # 更新 XML 路徑: 強制使用相對路徑 "../Import_mjcf/xxx"
            # 因為 XML 會放在 scene/ 資料夾，而資源在 Import_mjcf/
            new_path = "../Import_mjcf/" + rel_in_import.replace("\\", "/")
            node.set(attr_name, new_path)
            
        else:
            # === Case B: 檔案在外部 (例如 defaultscene 的資源) ===
            # 強制複製到 Import_mjcf/extras 資料夾
            fname = os.path.basename(abs_source_path)
            dest_path = os.path.join(target_import_dir, "extras", fname)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(abs_source_path, dest_path)
            
            # 更新 XML 路徑
            node.set(attr_name, f"../Import_mjcf/extras/{fname}")

    # 3. 掃描 XML 並處理路徑
    for tag in ["mesh", "texture", "hfield", "skin"]:
        for n in root.findall(f".//{tag}"):
            process_file_path(n)
    
    # 4. 寫入新的 XML
    target_xml_path = os.path.join(target_scene_dir, f"{scene_name}.xml")
    if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
    tree.write(target_xml_path, encoding="utf-8", xml_declaration=True)

    # 5. 打包 ZIP
    zip_filename = f"{scene_name}.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    
    print(f"[Export] Zipping to {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root_dir, dirs, files in os.walk(temp_root):
            for file in files:
                abs_file = os.path.join(root_dir, file)
                # 計算 ZIP 內部的相對路徑 (去掉 temp_export 前綴)
                rel_archive = os.path.relpath(abs_file, temp_root)
                zf.write(abs_file, rel_archive)

    # 6. 清理暫存
    shutil.rmtree(temp_root)
    print(f"[Export] Success!")
    return zip_path
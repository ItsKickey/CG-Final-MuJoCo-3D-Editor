# src/loader.py
import xml.etree.ElementTree as ET
import mujoco
import shutil
import os
from pathlib import Path

LAST_IMPORTED_BODY_NAME = None

def get_relative_path(target_path, base_dir):
    try:
        target_abs = os.path.abspath(target_path)
        base_abs = os.path.abspath(base_dir)
        rel_path = os.path.relpath(target_abs, base_abs)
        return rel_path.replace("\\", "/")
    except ValueError:
        return target_path.replace("\\", "/")

def load_scene_with_object(main_xml_path, obj_xml_path, spawn_height=5.0, save_merged_xml=None):
    global LAST_IMPORTED_BODY_NAME

    print(f"\n[loader] Processing: {os.path.basename(obj_xml_path)}")

    main_tree = ET.parse(main_xml_path)
    main_root = main_tree.getroot()
    main_worldbody = main_root.find("worldbody")

    main_xml_dir = os.path.dirname(os.path.abspath(main_xml_path))

    obj_xml_path = os.path.abspath(obj_xml_path)
    obj_dir = os.path.dirname(obj_xml_path)
    
    try:
        obj_tree = ET.parse(obj_xml_path)
    except FileNotFoundError:
        print(f"[loader] Error: File not found {obj_xml_path}")
        return None
        
    obj_root = obj_tree.getroot()
    obj_worldbody = obj_root.find("worldbody")

    class_defaults = {}
    for d in obj_root.findall(".//default"):
        cls = d.get("class")
        if cls:
            geom_node = d.find("geom")
            if geom_node is not None:
                class_defaults[cls] = geom_node.attrib

    def fix_asset_path(node, attr_name):
        filename = node.get(attr_name)
        if not filename: return
        abs_path = os.path.join(obj_dir, filename)
        if not os.path.exists(abs_path):
            parent_dir = os.path.dirname(obj_dir)
            alt_path = os.path.join(parent_dir, filename)
            if os.path.exists(alt_path): abs_path = alt_path
        rel_path = get_relative_path(abs_path, main_xml_dir)
        node.set(attr_name, rel_path)

    for mesh in obj_root.findall(".//mesh"): fix_asset_path(mesh, "file")
    for tex in obj_root.findall(".//texture"): fix_asset_path(tex, "file")

    main_asset = main_root.find("asset")
    if main_asset is None: main_asset = ET.SubElement(main_root, "asset")
    
    obj_asset = obj_root.find("asset")
    asset_remap = {}

    if obj_asset is not None:
        for item in obj_asset:
            original_name = item.get("name")
            item_tag = item.tag
            
            if item_tag == "mesh":
                base_name = original_name if original_name else Path(item.get("file")).stem
                unique_name = base_name
                k = 1
                while any(m.get("name") == unique_name for m in main_asset.findall("mesh")):
                    unique_name = f"{base_name}_{k}"
                    k += 1
                
                item.set("name", unique_name)
                main_asset.append(item)
                
                if original_name: asset_remap[original_name] = unique_name
                file_stem = Path(item.get("file")).stem
                asset_remap[file_stem] = unique_name
            else:
                exists = False
                if original_name:
                    for existing in main_asset:
                        if existing.get("name") == original_name and existing.tag == item_tag:
                            exists = True
                            break
                if not exists: main_asset.append(item)

    bodies = list(obj_worldbody)
    if not bodies: return None
    src_body = bodies[0]

    base_name = src_body.get("name", "imported")
    existing_names = set()
    for b in main_root.findall(".//body"):
        n = b.get("name")
        if n: existing_names.add(n)
    
    new_name = base_name
    k = 1
    while new_name in existing_names:
        new_name = f"{base_name}_{k}"
        k += 1
    src_body.set("name", new_name)
    LAST_IMPORTED_BODY_NAME = new_name
    
    src_body.set("pos", f"0 0 {spawn_height}")

    joint_node = src_body.find("joint")
    if joint_node is None:
        ET.SubElement(src_body, "joint", {"type": "free", "name": f"{new_name}_freejoint"})
    else:
        if not joint_node.get("name"):
            joint_node.set("name", f"{new_name}_freejoint")

    for g in src_body.findall(".//geom"):
        if "class" in g.attrib:
            cls_name = g.attrib["class"]
            if cls_name in class_defaults:
                for key, val in class_defaults[cls_name].items():
                    if key not in g.attrib: g.set(key, val)
            del g.attrib["class"]
        
        g.set("contype", "1"); g.set("conaffinity", "1")
        
        mesh_ref = g.get("mesh")
        if mesh_ref:
            if mesh_ref in asset_remap:
                g.set("mesh", asset_remap[mesh_ref])
            g.set("type", "mesh")
            if "material" not in g.attrib and "rgba" not in g.attrib:
                g.set("rgba", "0.9 0.9 0.9 1")

    main_worldbody.append(src_body)

    if main_asset is not None:
        for mesh in main_asset.findall("mesh"):
            if not mesh.get("name") and mesh.get("file"):
                gen_name = Path(mesh.get("file")).stem
                mesh.set("name", gen_name)

    if save_merged_xml:
        tree = ET.ElementTree(main_root)
        if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
        tree.write(save_merged_xml, encoding="utf-8", xml_declaration=True)

    xml_str = ET.tostring(main_root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_str)

def delete_body_from_scene(xml_path, body_name):
    """
    [修正] 安全刪除: 只有當 Mesh 或 Material 沒有被其他 Body 使用時，才將其刪除。
    防止刪除共用資源導致崩潰。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        asset = root.find("asset")
        
        target_body = None
        for body in worldbody.findall("body"):
            if body.get("name") == body_name:
                target_body = body
                break
        
        if target_body is None: return False

        # 1. 找出「目標 Body」使用的資源
        target_meshes = set()
        target_materials = set()
        
        for geom in target_body.findall(".//geom"):
            m = geom.get("mesh")
            if m: target_meshes.add(m)
            mat = geom.get("material")
            if mat: target_materials.add(mat)

        # 2. 找出「其他所有 Body」正在使用的資源 (白名單)
        keep_meshes = set()
        keep_materials = set()
        
        for body in worldbody.findall("body"):
            if body is target_body: continue # 跳過自己
            
            for geom in body.findall(".//geom"):
                m = geom.get("mesh")
                if m: keep_meshes.add(m)
                mat = geom.get("material")
                if mat: keep_materials.add(mat)

        # 3. 計算差集：只刪除「目標有用」且「其他人沒用」的資源
        meshes_to_delete = target_meshes - keep_meshes
        materials_to_delete = target_materials - keep_materials

        # 4. 執行刪除
        worldbody.remove(target_body)
        print(f"[loader] Deleted body: {body_name}")

        if asset is not None:
            # 刪除 Mesh
            count_m = 0
            for m_node in list(asset.findall("mesh")): # 用 list() 複製一份以便遍歷時刪除
                if m_node.get("name") in meshes_to_delete:
                    asset.remove(m_node)
                    count_m += 1
            if count_m > 0: print(f"[loader] Cleaned up {count_m} unused meshes.")

            # 刪除 Material
            count_mat = 0
            for mat_node in list(asset.findall("material")):
                if mat_node.get("name") in materials_to_delete:
                    asset.remove(mat_node)
                    count_mat += 1
            if count_mat > 0: print(f"[loader] Cleaned up {count_mat} unused materials.")

        if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as e:
        print(f"[loader] Delete Error: {e}")
        return False

def update_body_xml(xml_path, body_name, pos, quat):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        found = False
        for body in worldbody.findall("body"):
            if body.get("name") == body_name:
                pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
                quat_str = f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}"
                body.set("pos", pos_str)
                body.set("quat", quat_str)
                found = True
                break
        if found:
            if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return True
        return False
    except Exception as e:
        print(f"[loader] Update XML Error: {e}")
        return False
    
# [新增] 批次更新所有物體的位置 (用於 Auto-Sync)
def batch_update_bodies_xml(xml_path, model, data):
    """
    遍歷模型中所有自由移動的 Body，將其當前的 qpos (位置/旋轉) 寫入 XML。
    這用於在 Import 或其他 Reload 操作前，保存物理模擬的結果。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        
        updated_count = 0
        
        # 建立 XML 中 Body Name -> Body Node 的映射，加速搜尋
        xml_bodies = {b.get("name"): b for b in worldbody.findall("body") if b.get("name")}

        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or name not in xml_bodies: continue

            # 檢查是否有 Free Joint (只有自由物體需要更新位置，靜態物體不動)
            jnt_adr = model.body_jntadr[i]
            if jnt_adr >= 0 and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = model.jnt_qposadr[jnt_adr]
                
                # 讀取當前物理狀態
                pos = data.qpos[qpos_adr : qpos_adr+3]
                quat = data.qpos[qpos_adr+3 : qpos_adr+7]
                
                # 寫入 XML 節點
                body_node = xml_bodies[name]
                body_node.set("pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
                body_node.set("quat", f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}")
                updated_count += 1
        
        if updated_count > 0:
            if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            # print(f"[loader] Auto-Saved {updated_count} bodies to XML.")
            return True
        return False
    except Exception as e:
        print(f"[loader] Batch Update Error: {e}")
        return False

def add_light_to_scene(xml_path, spawn_pos="0 0 3"):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        
        base_name = "point_light"
        unique_name = base_name
        k = 1
        while any(b.get("name") == unique_name for b in worldbody.findall("body")):
            unique_name = f"{base_name}_{k}"
            k += 1
            
        print(f"[loader] Creating light: {unique_name}")
        
        light_body = ET.SubElement(worldbody, "body", {
            "name": unique_name,
            "pos": spawn_pos
        })
        ET.SubElement(light_body, "joint", {"type": "free", "name": f"{unique_name}_joint"})
        ET.SubElement(light_body, "light", {
            "mode": "trackcom",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.3 0.3 0.3",
            "castshadow": "true",
            "dir": "0 0 -1"
        })
        ET.SubElement(light_body, "geom", {
            "type": "sphere",
            "size": "0.1",
            "rgba": "1 1 0.8 0.3",
            "contype": "0",
            "conaffinity": "0",
            "group": "1"
        })
        
        global LAST_IMPORTED_BODY_NAME
        LAST_IMPORTED_BODY_NAME = unique_name
        
        if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as e:
        print(f"[loader] Add Light Error: {e}")
        return False

def update_light_xml(xml_path, body_name, rgb):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        found = False
        for body in worldbody.findall("body"):
            if body.get("name") == body_name:
                light = body.find("light")
                if light is not None:
                    rgb_str = f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}"
                    light.set("diffuse", rgb_str)
                    light.set("specular", f"{rgb[0]*0.5:.3f} {rgb[1]*0.5:.3f} {rgb[2]*0.5:.3f}")
                    geom = body.find("geom")
                    if geom is not None:
                        geom.set("rgba", f"{rgb[0]} {rgb[1]} {rgb[2]} 0.8")
                    found = True
                break
        if found:
            if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return True
        return False
    except Exception as e:
        print(f"[loader] Update Light Error: {e}")
        return False

def change_floor_texture(xml_path, image_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        asset = root.find("asset")
        if asset is None: return False
        
        target_tex = None
        for tex in asset.findall("texture"):
            if tex.get("name") == "grid":
                target_tex = tex
                break
                
        if target_tex is not None:
            # 清除舊的生成屬性
            for attr in ["builtin", "rgb1", "rgb2", "mark", "markrgb", "width", "height"]:
                if attr in target_tex.attrib:
                    del target_tex.attrib[attr]
            
            target_tex.set("type", "2d")
            
            project_root = os.getcwd()
            dest_dir = os.path.join(project_root, "Import_mjcf", "grid_textures")
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir) 
            
            filename = os.path.basename(image_path)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                if os.path.abspath(image_path) != os.path.abspath(dest_path):
                    shutil.copy2(image_path, dest_path)
                    print(f"[loader] Copied texture to: {dest_path}")
            except Exception as e:
                print(f"[loader] Warning: Copy failed ({e}). Using original path.")
                dest_path = image_path 

            xml_dir = os.path.dirname(os.path.abspath(xml_path))
            rel_path = get_relative_path(dest_path, xml_dir)
            target_tex.set("file", rel_path)
            
            if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"[loader] Floor texture changed to: {rel_path}")
            return True
        else:
            print("[loader] Error: Texture 'grid' not found in asset.")
            return False
            
    except Exception as e:
        print(f"[loader] Change Floor Error: {e}")
        return False
# src/loader.py (V18 Update Logic)
import xml.etree.ElementTree as ET
import mujoco
import os
from pathlib import Path

LAST_IMPORTED_BODY_NAME = None

def load_scene_with_object(main_xml_path, obj_xml_path, spawn_height=5.0, save_merged_xml=None):
    global LAST_IMPORTED_BODY_NAME

    print(f"\n[loader] Processing: {os.path.basename(obj_xml_path)}")

    # 1. 讀取主場景
    main_tree = ET.parse(main_xml_path)
    main_root = main_tree.getroot()
    main_worldbody = main_root.find("worldbody")

    # 2. 讀取物件 MJCF
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
        node.set(attr_name, abs_path.replace("\\", "/"))

    for mesh in obj_root.findall(".//mesh"): fix_asset_path(mesh, "file")
    for tex in obj_root.findall(".//texture"): fix_asset_path(tex, "file")

    # 3. Assets
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

    # 4. Body
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

    # 5. Global Asset Fix (V16)
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
    """從 XML 移除 Body"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        found = False
        for body in worldbody.findall("body"):
            if body.get("name") == body_name:
                worldbody.remove(body)
                found = True
                print(f"[loader] Deleted body: {body_name}")
                break
        if found:
            if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return True
        return False
    except Exception as e:
        print(f"[loader] Delete Error: {e}")
        return False

# ==== [新功能] 更新 Body 座標回 XML ====
def update_body_xml(xml_path, body_name, pos, quat):
    """將物體的最終位置寫入 XML，以便存檔與 Undo"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        
        found = False
        for body in worldbody.findall("body"):
            if body.get("name") == body_name:
                # 格式化為字串 "x y z"
                pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
                quat_str = f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}"
                
                body.set("pos", pos_str)
                body.set("quat", quat_str)
                found = True
                break
        
        if found:
            if hasattr(ET, "indent"): ET.indent(tree, space="  ", level=0)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            # print(f"[loader] Updated XML pose for {body_name}")
            return True
        return False
    except Exception as e:
        print(f"[loader] Update XML Error: {e}")
        return False
# src/utils.py
import numpy as np
import mujoco

def euler2quat(r, p, y):
    """將歐拉角 (度) 轉換為四元數 [w, x, y, z]"""
    r, p, y = np.radians([r, p, y])
    cr, sr = np.cos(r*0.5), np.sin(r*0.5)
    cp, sp = np.cos(p*0.5), np.sin(p*0.5)
    cy, sy = np.cos(y*0.5), np.sin(y*0.5)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

def quat2euler(quat):
    """將四元數 [w, x, y, z] 轉換為歐拉角 (度)"""
    w, x, y, z = quat
    sinr_cosp = 2*(w*x + y*z); cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(sinp) if abs(sinp) < 1 else np.copysign(np.pi/2, sinp)
    siny_cosp = 2*(w*z + x*y); cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])

def get_body_geoms(model, body_id):
    """回傳屬於該 Body 的所有 geom_id"""
    geoms = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == body_id:
            geoms.append(i)
    return geoms

def save_simulation_state(model, data):
    """保存當前物理狀態"""
    if model is None or data is None: return {}
    saved = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_adr = model.jnt_qposadr[i]; qvel_adr = model.jnt_dofadr[i]
            jnt_type = model.jnt_type[i]
            q_len = 7 if jnt_type == mujoco.mjtJoint.mjJNT_FREE else 1
            v_len = 6 if jnt_type == mujoco.mjtJoint.mjJNT_FREE else 1
            saved[name] = {"qpos": data.qpos[qpos_adr:qpos_adr+q_len].copy(), 
                           "qvel": data.qvel[qvel_adr:qvel_adr+v_len].copy()}
    return saved

def restore_simulation_state(model, data, saved_state):
    """還原物理狀態"""
    if not saved_state: return
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name in saved_state:
            vals = saved_state[name]
            qpos_adr = model.jnt_qposadr[i]; qvel_adr = model.jnt_dofadr[i]
            q_len = len(vals["qpos"]); v_len = len(vals["qvel"])
            data.qpos[qpos_adr:qpos_adr+q_len] = vals["qpos"]
            data.qvel[qvel_adr:qvel_adr+v_len] = vals["qvel"]
    mujoco.mj_forward(model, data)
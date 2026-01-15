import mujoco
import mujoco.viewer
import numpy as np
import os

# --- 1. 配置路径（请根据你的实际文件名修改） ---
MODEL_PATH = '/anyverse/sub_modules/isaacros/src/wheeled_humanoid/robot_model/urdf/wheel_robot.urdf'  # 你的 MuJoCo 模型文件
LEFT_PTS_PATH = 'left_reachable.npy'
RIGHT_PTS_PATH = 'right_reachable.npy'

def load_points(path):
    if os.path.exists(path):
        data = np.load(path)
        print(f"成功加载 {path}: {data.shape[0]} 个点")
        return data
    else:
        print(f"警告: 找不到文件 {path}")
        return np.array([])

def main():
    # --- 2. 加载模型与数据 ---
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    left_pts = load_points(LEFT_PTS_PATH)
    right_pts = load_points(RIGHT_PTS_PATH)

    # --- 3. 启动交互式仿真器 ---
    # 使用 launch_passive 允许我们在循环中动态添加可视化元素
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo 仿真器已启动。")
        
        while viewer.is_running():
            # 重置当前帧的自定义几何体计数
            viewer.user_scn.ngeom = 0
            
            # --- 4. 渲染左臂点云 (红色) ---
            for p in left_pts:
                # 检查点云是否超过 MuJoCo 默认的最大几何体限制 (通常为 1000)
                if viewer.user_scn.ngeom >= 10000: break 
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],   # 点的大小
                    pos=p,                # 点的坐标
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 0.6]   # 红色，60% 透明度
                )
                viewer.user_scn.ngeom += 1

            # --- 5. 渲染右臂点云 (蓝色) ---
            for p in right_pts:
                if viewer.user_scn.ngeom >= 20000: break
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=p,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 0, 1, 0.6]   # 蓝色，60% 透明度
                )
                viewer.user_scn.ngeom += 1

            # 推进仿真步进（如果不需要物理运动，可以注释掉下面两行）
            # mujoco.mj_step(model, data)
            
            # 刷新查看器
            viewer.sync()

if __name__ == "__main__":
    main()
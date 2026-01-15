import os
import pinocchio as pin
import numpy as np

# ================= 配置区 =================
# 请确保这个路径是正确的！
URDF_PATH = "/anyverse/sub_modules/isaacros/src/wheeled_humanoid/robot_model/urdf/wheel_robot.urdf"

# 辅助函数：推测 package 路径
def guess_package_dirs(urdf_path):
    dirs = []
    urdf_dir = os.path.dirname(urdf_path)
    dirs.append(urdf_dir)
    cur = urdf_dir
    for _ in range(5):
        cur = os.path.dirname(cur)
        if cur and cur not in dirs:
            dirs.append(cur)
    dirs.append("/anyverse") # 添加你的根目录
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]

# ================= 核心诊断逻辑 =================
def diagnose():
    print(f"🔍 正在加载 URDF: {URDF_PATH}")
    package_dirs = guess_package_dirs(URDF_PATH)
    
    # 1. 加载模型 (Collision)
    try:
        model = pin.buildModelFromUrdf(URDF_PATH)
        col_model = pin.buildGeomFromUrdf(model, URDF_PATH, pin.GeometryType.COLLISION, package_dirs)
        col_data = col_model.createData()
        data = model.createData()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 开启全量碰撞检测
    col_model.addAllCollisionPairs()
    print(f"📦 原始碰撞对总数: {len(col_model.collisionPairs)}")

    # 3. 设置标准姿态
    q_ref = pin.neutral(model)
    
    # 4. 计算碰撞
    pin.computeCollisions(model, data, col_model, col_data, q_ref, False)

    # 5. 打印碰撞结果
    collision_count = 0
    print("-" * 30)
    print("🛑 初始姿态碰撞详情 (Top 10):")
    
    for k in range(len(col_model.collisionPairs)):
        if col_data.collisionResults[k].isCollision():
            collision_count += 1
            pair = col_model.collisionPairs[k]
            name1 = col_model.geometryObjects[pair.first].name
            name2 = col_model.geometryObjects[pair.second].name
            
            # 检查是否是相邻关节
            joint1 = col_model.geometryObjects[pair.first].parentJoint
            joint2 = col_model.geometryObjects[pair.second].parentJoint
            
            is_adjacent = (joint1 == joint2) or \
                          (model.parents[joint2] == joint1) or \
                          (model.parents[joint1] == joint2)
            
            relation = "相邻(父子)" if is_adjacent else "非相邻"
            
            if collision_count <= 10:
                print(f"  [{collision_count}] {name1} <--> {name2} | 关系: {relation}")

    print("-" * 30)
    print(f"📊 结论: 共发现 {collision_count} 处碰撞。")
    
    if collision_count > 0:
        print("💡 这是一个【假阳性】问题！因为 Pinocchio 默认不会忽略相邻连杆。")
        print("✅ 解决方法：请使用我刚才给你的 'init_worker' 代码，它包含自动移除逻辑。")
    else:
        print("✅ 神奇，竟然没有碰撞。那 IK 结果为 0 可能是目标姿态 R_d 设置得太难了。")

if __name__ == "__main__":
    diagnose()
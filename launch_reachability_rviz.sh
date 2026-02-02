#!/bin/bash
# ============================================================================
# 机械臂可达空间分析 + RViz2 可视化 一键启动脚本
# ============================================================================
# 用法:
#   ./launch_reachability_rviz.sh <URDF路径> [选项]
#
# 示例:
#   ./launch_reachability_rviz.sh robot_model/urdf/wheel_robot.urdf --arm both
#   ./launch_reachability_rviz.sh robot_model/urdf/wheel_robot.urdf --arm both --skip-analysis
# ============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 默认值
URDF_PATH=""
BASE_LINK=""
EE_LINK=""
RESOLUTION=0.05
ARM_MODE="auto"
ARM_NAME=""
SKIP_ANALYSIS=false
CLEAN_ROS=false
KILL_RVIZ=false
POINTS_FRAME="world"
TRANSLATION_ONLY=false
PUBLISH_TF=false
CLEAN_ROS=false
KILL_RVIZ=false
OUTPUT_DIR=""
PREFIX="reachability"
AUTO_BOUNDS=true
X_RANGE_MIN=""
X_RANGE_MAX=""
Y_RANGE_MIN=""
Y_RANGE_MAX=""
Z_RANGE_MIN=""
Z_RANGE_MAX=""
NUM_SEEDS=32
BATCH_SIZE=1024
POS_THRESHOLD=0.005
ROT_THRESHOLD=0.05
ORIENTATION_MODE="multi_fixed"
NUM_ORIENTATIONS=6
DEVICE="cuda:0"
NO_MANIPULABILITY=false
CONFIG_PATH=""
RVIZ_FRAME="world"
RVIZ_TOPIC="/reachability/points"
RVIZ_RATE=1.0
RVIZ_ONCE=false
RVIZ_COLOR_MODE="dexterity"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    echo "用法: $0 <URDF路径> [选项]"
    echo ""
    echo "选项:"
    echo "  --arm MODE          分析模式: auto/left/right/both/all (默认: auto)"
    echo "  --arm-name NAME     指定单臂名称（配合 --arm single）"
    echo "  --base-link NAME    基座链接名"
    echo "  --ee-link NAME      末端执行器链接名"
    echo "  --resolution FLOAT  体素分辨率 (默认: 0.05)"
    echo "  --points-frame F    点云坐标系: base/world (默认: world)"
    echo "  --x-range MIN MAX   X 轴范围"
    echo "  --y-range MIN MAX   Y 轴范围"
    echo "  --z-range MIN MAX   Z 轴范围"
    echo "  --no-auto-bounds    禁用自动检测工作空间边界"
    echo "  --num-seeds N       IK 种子数量"
    echo "  --batch-size N      IK 批处理大小"
    echo "  --pos-threshold V   IK 位置误差阈值"
    echo "  --rot-threshold V   IK 旋转误差阈值"
    echo "  --orientation-mode M 姿态模式: fixed/multi_fixed/spherical/random"
    echo "  --num-orientations N 姿态采样数量（spherical/random）"
    echo "  --device DEV        计算设备 (默认: cuda:0)"
    echo "  --no-manipulability 跳过可操作度计算"
    echo "  --output DIR        输出目录（默认: reachability_output_<URDF>）"
    echo "  --prefix NAME       输出文件前缀（默认: reachability）"
    echo "  --config PATH       YAML 配置文件路径"
    echo "  --translation-only  发布时仅平移，不应用 base_transform 旋转"
    echo "  --publish-tf        使用点云发布器发布 TF（将不启动 robot_state_publisher）"
    echo "  --rviz-color-mode M RViz 颜色模式: dexterity/arm/tint (默认: dexterity)"
    echo "  --rviz-frame F      RViz Fixed Frame (默认: world)"
    echo "  --rviz-topic T      RViz 点云 Topic (如 /reachability/points)"
    echo "  --rviz-rate Hz      RViz 发布频率 (0 表示只发布一次)"
    echo "  --rviz-once         RViz 仅发布一次（等同 --rviz-rate 0）"
    echo "  --skip-analysis     跳过可达性分析，直接可视化已有数据"
    echo "  --clean-ros         清理 ROS2 缓存并重启 daemon（会清空 ~/.ros）"
    echo "  --kill-rviz         启动前先关闭已有 rviz2"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  # 双臂分析 + 可视化"
    echo "  $0 robot_model/urdf/wheel_robot.urdf --arm both"
    echo ""
    echo "  # 跳过分析，仅可视化"
    echo "  $0 robot_model/urdf/wheel_robot.urdf --arm both --skip-analysis"
}

# 解析参数
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

URDF_PATH="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --arm) ARM_MODE="$2"; shift 2 ;;
        --arm-name) ARM_NAME="$2"; shift 2 ;;
        --base-link) BASE_LINK="$2"; shift 2 ;;
        --ee-link) EE_LINK="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --points-frame) POINTS_FRAME="$2"; shift 2 ;;
        --x-range) X_RANGE_MIN="$2"; X_RANGE_MAX="$3"; shift 3 ;;
        --y-range) Y_RANGE_MIN="$2"; Y_RANGE_MAX="$3"; shift 3 ;;
        --z-range) Z_RANGE_MIN="$2"; Z_RANGE_MAX="$3"; shift 3 ;;
        --no-auto-bounds) AUTO_BOUNDS=false; shift ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --pos-threshold) POS_THRESHOLD="$2"; shift 2 ;;
        --rot-threshold) ROT_THRESHOLD="$2"; shift 2 ;;
        --orientation-mode) ORIENTATION_MODE="$2"; shift 2 ;;
        --num-orientations) NUM_ORIENTATIONS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --no-manipulability) NO_MANIPULABILITY=true; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --translation-only) TRANSLATION_ONLY=true; shift ;;
        --publish-tf) PUBLISH_TF=true; shift ;;
        --rviz-color-mode|--color-mode) RVIZ_COLOR_MODE="$2"; shift 2 ;;
        --rviz-frame) RVIZ_FRAME="$2"; shift 2 ;;
        --rviz-topic) RVIZ_TOPIC="$2"; shift 2 ;;
        --rviz-rate) RVIZ_RATE="$2"; shift 2 ;;
        --rviz-once) RVIZ_ONCE=true; shift ;;
        --skip-analysis) SKIP_ANALYSIS=true; shift ;;
        --clean-ros) CLEAN_ROS=true; shift ;;
        --kill-rviz) KILL_RVIZ=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) print_error "未知参数: $1" ;;
    esac
done

# 验证 URDF
if [ ! -f "$URDF_PATH" ]; then
    print_error "URDF 文件不存在: $URDF_PATH"
fi

URDF_ABS=$(realpath "$URDF_PATH")
URDF_DIR=$(dirname "$URDF_ABS")
URDF_BASENAME=$(basename "$URDF_ABS" .urdf)

# 默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$SCRIPT_DIR/reachability_output_${URDF_BASENAME}"
fi

# RViz 发布频率：若指定一次发布，则强制 0
if [ "$RVIZ_ONCE" = true ]; then
    RVIZ_RATE=0.0
fi

# RViz topic 处理：允许传入 /reachability/points，自动转换为 base topic
RVIZ_BASE_TOPIC="$RVIZ_TOPIC"
if [[ "$RVIZ_BASE_TOPIC" != /* ]]; then
    RVIZ_BASE_TOPIC="/$RVIZ_BASE_TOPIC"
fi
if [[ "$RVIZ_BASE_TOPIC" == */* ]]; then
    LAST_SEG="${RVIZ_BASE_TOPIC##*/}"
    BASE_SEG="${RVIZ_BASE_TOPIC%/*}"
    if [[ "$LAST_SEG" == points* ]]; then
        RVIZ_BASE_TOPIC="$BASE_SEG"
        if [ -z "$RVIZ_BASE_TOPIC" ]; then
            RVIZ_BASE_TOPIC="/"
        fi
    fi
fi

# 自动检测 ROS 包目录 (用于解析 package:// 路径)
# URDF 结构通常是: package_name/urdf/robot.urdf
PACKAGE_DIR=$(dirname "$URDF_DIR")
PACKAGE_PARENT=$(dirname "$PACKAGE_DIR")

echo "============================================================"
echo "机械臂可达空间分析 + RViz2 可视化"
echo "============================================================"
echo "URDF: $URDF_ABS"
echo "包目录: $PACKAGE_DIR"
echo "分析模式: $ARM_MODE"
echo "跳过分析: $SKIP_ANALYSIS"
echo "输出目录: $OUTPUT_DIR"
echo "RViz Fixed Frame: $RVIZ_FRAME"
echo "RViz Topic: $RVIZ_TOPIC"
echo "RViz 颜色模式: $RVIZ_COLOR_MODE"
echo "============================================================"

# 确保 ROS2 环境
source /opt/ros/humble/setup.bash

# 设置 AMENT_PREFIX_PATH 以解析 package:// 路径
export AMENT_PREFIX_PATH="$PACKAGE_PARENT:$AMENT_PREFIX_PATH"
print_info "AMENT_PREFIX_PATH: $AMENT_PREFIX_PATH"

# Step 1: 清理旧进程
print_info "步骤 1: 清理旧的 ROS 节点..."
pkill -9 -f robot_state_publisher 2>/dev/null || true
pkill -9 -f joint_state_publisher 2>/dev/null || true
pkill -9 -f rviz_publisher_node 2>/dev/null || true
pkill -9 -f reachability_visualizer 2>/dev/null || true
if [ "$CLEAN_ROS" = true ]; then
    KILL_RVIZ=true
    print_info "清理 ROS2 缓存并重启 daemon..."
    ros2 daemon stop 2>/dev/null || true
    rm -rf ~/.ros
    ros2 daemon start 2>/dev/null || true
    print_success "ROS2 daemon 已重启，缓存已清理"
fi
if [ "$KILL_RVIZ" = true ]; then
    print_info "关闭已有 rviz2..."
    pkill -9 -f rviz2 2>/dev/null || true
fi
sleep 2
print_success "旧进程已清理"

# 计算并打印点云 world 坐标范围（基于保存的数据）
print_pointcloud_range() {
    local name="$1"
    local data_dir="$2"
    local prefix="$3"
    python3 << PYEOF
import os, json, numpy as np

name = "${name}"
data_dir = "${data_dir}"
prefix = "${prefix}"

stats_path = os.path.join(data_dir, f"{prefix}_stats.json")
grid_path = os.path.join(data_dir, f"{prefix}_grid_points.npy")
mask_path = os.path.join(data_dir, f"{prefix}_reachable_mask.npy")

if not (os.path.exists(stats_path) and os.path.exists(grid_path) and os.path.exists(mask_path)):
    print(f"[RANGE][WARN] {name}: 缺少数据文件，跳过范围计算")
    raise SystemExit(0)

with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)

points_frame = stats.get("points_frame", "base")

grid = np.load(grid_path)
mask = np.load(mask_path)
pts = grid[mask]

if pts.size == 0:
    print(f"[RANGE] {name}: 可达点为 0")
    raise SystemExit(0)

if points_frame != "world":
    bt = stats.get("base_transform")
    if bt and "rotation" in bt and "translation" in bt:
        R = np.array(bt["rotation"], dtype=float)
        t = np.array(bt["translation"], dtype=float).reshape(3)
        pts = (R @ pts.T).T + t
    else:
        print(f"[RANGE][WARN] {name}: 无 base_transform，按 base 视为 world")

mn = pts.min(axis=0)
mx = pts.max(axis=0)
print(f"[RANGE] {name}: X[{mn[0]:.3f},{mx[0]:.3f}] "
      f"Y[{mn[1]:.3f},{mx[1]:.3f}] Z[{mn[2]:.3f},{mx[2]:.3f}] "
      f"(points_frame={points_frame})")
PYEOF
}

# Step 2: 设置输出目录
print_info "输出目录: $OUTPUT_DIR"

# Step 3: 运行可达性分析（如果需要）
if [ "$SKIP_ANALYSIS" = false ]; then
    print_info "步骤 2: 运行可达性分析..."

    # 构建命令参数
    ANALYSIS_ARGS="--urdf '$URDF_ABS' --arm $ARM_MODE --resolution $RESOLUTION --points-frame $POINTS_FRAME --output '$OUTPUT_DIR' --prefix '$PREFIX' --no-visualization"
    if [ -n "$ARM_NAME" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --arm-name '$ARM_NAME'"
    fi
    if [ -n "$BASE_LINK" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --base-link '$BASE_LINK'"
    fi
    if [ -n "$EE_LINK" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --ee-link '$EE_LINK'"
    fi
    if [ -n "$X_RANGE_MIN" ] && [ -n "$X_RANGE_MAX" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --x-range $X_RANGE_MIN $X_RANGE_MAX"
    fi
    if [ -n "$Y_RANGE_MIN" ] && [ -n "$Y_RANGE_MAX" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --y-range $Y_RANGE_MIN $Y_RANGE_MAX"
    fi
    if [ -n "$Z_RANGE_MIN" ] && [ -n "$Z_RANGE_MAX" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --z-range $Z_RANGE_MIN $Z_RANGE_MAX"
    fi
    if [ "$AUTO_BOUNDS" = false ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --no-auto-bounds"
    fi
    ANALYSIS_ARGS="$ANALYSIS_ARGS --num-seeds $NUM_SEEDS --batch-size $BATCH_SIZE --pos-threshold $POS_THRESHOLD --rot-threshold $ROT_THRESHOLD"
    ANALYSIS_ARGS="$ANALYSIS_ARGS --orientation-mode $ORIENTATION_MODE --num-orientations $NUM_ORIENTATIONS"
    ANALYSIS_ARGS="$ANALYSIS_ARGS --device $DEVICE"
    if [ "$NO_MANIPULABILITY" = true ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --no-manipulability"
    fi
    if [ -n "$CONFIG_PATH" ]; then
        ANALYSIS_ARGS="$ANALYSIS_ARGS --config '$CONFIG_PATH'"
    fi

    cd "$SCRIPT_DIR"
    eval "python3 main.py $ANALYSIS_ARGS"

    if [ $? -eq 0 ]; then
        print_success "可达性分析完成"
    else
        print_error "可达性分析失败"
    fi
else
    print_warning "跳过可达性分析"
fi

# Step 4: 启动 ROS 节点
print_info "步骤 3: 启动 ROS 节点..."

if [ "$PUBLISH_TF" = true ]; then
    print_warning "将由点云发布器发布 TF，跳过 robot_state_publisher/joint_state_publisher"
else
    # 使用 xacro 或直接读取 URDF，生成参数文件
    PARAM_FILE="/tmp/robot_state_publisher_params.yaml"
    python3 << PYEOF
import yaml

with open('$URDF_ABS', 'r') as f:
    urdf_content = f.read()

params = {
    'robot_state_publisher': {
        'ros__parameters': {
            'robot_description': urdf_content
        }
    }
}

with open('$PARAM_FILE', 'w') as f:
    yaml.dump(params, f, default_flow_style=False)

print('参数文件已生成: $PARAM_FILE')
PYEOF

    # 启动 robot_state_publisher (后台运行)
    print_info "启动 robot_state_publisher..."
    ros2 run robot_state_publisher robot_state_publisher \
        --ros-args --params-file "$PARAM_FILE" \
        > /tmp/rsp.log 2>&1 &
    RSP_PID=$!

    # 启动 joint_state_publisher (后台运行)
    print_info "启动 joint_state_publisher..."
    ros2 run joint_state_publisher joint_state_publisher \
        > /tmp/jsp.log 2>&1 &
    JSP_PID=$!

    sleep 3

    # 检查节点是否启动
    if ps -p $RSP_PID > /dev/null 2>&1; then
        print_success "robot_state_publisher 已启动 (PID: $RSP_PID)"
    else
        print_error "robot_state_publisher 启动失败，查看日志: /tmp/rsp.log"
    fi

    if ps -p $JSP_PID > /dev/null 2>&1; then
        print_success "joint_state_publisher 已启动 (PID: $JSP_PID)"
    else
        print_warning "joint_state_publisher 启动失败，查看日志: /tmp/jsp.log"
    fi
fi

# Step 5: 启动点云发布节点
print_info "步骤 4: 启动点云发布节点..."

cd "$SCRIPT_DIR"

if [ "$ARM_MODE" = "both" ] || [ "$ARM_MODE" = "all" ]; then
    # 双臂模式
    LEFT_DATA="$OUTPUT_DIR/left"
    RIGHT_DATA="$OUTPUT_DIR/right"

    if [ ! -d "$LEFT_DATA" ] || [ ! -d "$RIGHT_DATA" ]; then
        print_error "找不到双臂数据目录: $LEFT_DATA 或 $RIGHT_DATA"
    fi

    print_info "计算双臂点云 world 坐标范围..."
    print_pointcloud_range "left" "$LEFT_DATA" "reachability_left"
    print_pointcloud_range "right" "$RIGHT_DATA" "reachability_right"

    print_info "启动双臂点云发布..."
    TRANSFORM_FLAG=""
    if [ "$TRANSLATION_ONLY" = true ]; then
        TRANSFORM_FLAG="--translation-only"
    fi
    TF_FLAG=""
    if [ "$PUBLISH_TF" = false ]; then
        TF_FLAG="--no-tf"
    fi
    python3 arm_reachability/rviz_publisher_node.py \
        --arm both \
        --left-data "$LEFT_DATA" --left-prefix "reachability_left" \
        --right-data "$RIGHT_DATA" --right-prefix "reachability_right" \
        --urdf "$URDF_ABS" \
        --frame-id "$RVIZ_FRAME" \
        --topic "$RVIZ_BASE_TOPIC" \
        --rate "$RVIZ_RATE" \
        --color-mode "$RVIZ_COLOR_MODE" \
        $TRANSFORM_FLAG $TF_FLAG \
        
        > /tmp/rviz_pub.log 2>&1 &
    PUB_PID=$!

    sleep 2
    if ps -p $PUB_PID > /dev/null 2>&1; then
        print_success "双臂点云发布节点已启动 (PID: $PUB_PID)"
        print_info "Topics: /reachability/points_left, /reachability/points_right"
    else
        print_error "点云发布节点启动失败，查看日志: /tmp/rviz_pub.log"
    fi

elif [ "$ARM_MODE" = "left" ]; then
    # 仅左臂
    LEFT_DATA="$OUTPUT_DIR/left"
    if [ ! -d "$LEFT_DATA" ]; then
        LEFT_DATA="$OUTPUT_DIR"
    fi

    print_info "计算左臂点云 world 坐标范围..."
    print_pointcloud_range "left" "$LEFT_DATA" "reachability_left"

    TRANSFORM_FLAG=""
    if [ "$TRANSLATION_ONLY" = true ]; then
        TRANSFORM_FLAG="--translation-only"
    fi
    TF_FLAG=""
    if [ "$PUBLISH_TF" = false ]; then
        TF_FLAG="--no-tf"
    fi
    python3 arm_reachability/rviz_publisher_node.py \
        --arm left \
        --left-data "$LEFT_DATA" --left-prefix "reachability_left" \
        --urdf "$URDF_ABS" \
        --frame-id "$RVIZ_FRAME" \
        --topic "$RVIZ_BASE_TOPIC" \
        --rate "$RVIZ_RATE" \
        --color-mode "$RVIZ_COLOR_MODE" \
        $TRANSFORM_FLAG $TF_FLAG \
        
        > /tmp/rviz_pub.log 2>&1 &
    PUB_PID=$!

    sleep 2
    print_success "左臂点云发布节点已启动 (PID: $PUB_PID)"

elif [ "$ARM_MODE" = "right" ]; then
    # 仅右臂
    RIGHT_DATA="$OUTPUT_DIR/right"
    if [ ! -d "$RIGHT_DATA" ]; then
        RIGHT_DATA="$OUTPUT_DIR"
    fi

    print_info "计算右臂点云 world 坐标范围..."
    print_pointcloud_range "right" "$RIGHT_DATA" "reachability_right"

    TRANSFORM_FLAG=""
    if [ "$TRANSLATION_ONLY" = true ]; then
        TRANSFORM_FLAG="--translation-only"
    fi
    TF_FLAG=""
    if [ "$PUBLISH_TF" = false ]; then
        TF_FLAG="--no-tf"
    fi
    python3 arm_reachability/rviz_publisher_node.py \
        --arm right \
        --right-data "$RIGHT_DATA" --right-prefix "reachability_right" \
        --urdf "$URDF_ABS" \
        --frame-id "$RVIZ_FRAME" \
        --topic "$RVIZ_BASE_TOPIC" \
        --rate "$RVIZ_RATE" \
        --color-mode "$RVIZ_COLOR_MODE" \
        $TRANSFORM_FLAG $TF_FLAG \
        > /tmp/rviz_pub.log 2>&1 &
    PUB_PID=$!

    sleep 2
    print_success "右臂点云发布节点已启动 (PID: $PUB_PID)"

else
    # 单臂/自动模式
    print_info "计算点云 world 坐标范围..."
    print_pointcloud_range "arm" "$OUTPUT_DIR" "$PREFIX"

    TRANSFORM_FLAG=""
    if [ "$TRANSLATION_ONLY" = true ]; then
        TRANSFORM_FLAG="--translation-only"
    fi
    TF_FLAG=""
    if [ "$PUBLISH_TF" = false ]; then
        TF_FLAG="--no-tf"
    fi
    python3 arm_reachability/rviz_publisher_node.py \
        --arm single \
        --data-dir "$OUTPUT_DIR" --prefix "$PREFIX" \
        --urdf "$URDF_ABS" \
        --frame-id "$RVIZ_FRAME" \
        --topic "$RVIZ_BASE_TOPIC" \
        --rate "$RVIZ_RATE" \
        --color-mode "$RVIZ_COLOR_MODE" \
        $TRANSFORM_FLAG $TF_FLAG \
        > /tmp/rviz_pub.log 2>&1 &
    PUB_PID=$!

    sleep 2
    print_success "点云发布节点已启动 (PID: $PUB_PID)"
fi

# 清理函数
cleanup() {
    print_info "正在停止所有节点..."
    if [ "$PUBLISH_TF" = false ]; then
        kill $RSP_PID 2>/dev/null || true
        kill $JSP_PID 2>/dev/null || true
    fi
    kill $PUB_PID 2>/dev/null || true
    print_success "所有节点已停止"
}

trap cleanup EXIT INT TERM

# Step 6: 启动 RViz2
print_info "步骤 5: 启动 RViz2..."

echo ""
echo "============================================================"
print_success "所有节点已启动！"
echo ""
echo "正在启动 RViz2..."
echo ""
echo "在 RViz2 中配置:"
echo "  1. Fixed Frame: world"
echo "  2. Add -> RobotModel -> Description Topic: /robot_description"
if [ "$ARM_MODE" = "both" ] || [ "$ARM_MODE" = "all" ]; then
    echo "  3. Add -> PointCloud2 -> Topic: /reachability/points_left"
    echo "  4. Add -> PointCloud2 -> Topic: /reachability/points_right"
else
    echo "  3. Add -> PointCloud2 -> Topic: /reachability/points"
fi
echo ""
echo "按 Ctrl+C 停止所有节点"
echo "============================================================"

# 启动 RViz2 (前台运行)
rviz2

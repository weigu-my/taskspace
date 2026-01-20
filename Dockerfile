# =============================================================================
# 机械臂可达性分析框架 Docker 镜像
#
# 基于 NVIDIA CUDA 镜像，支持 cuRobo GPU 加速 IK 求解
# =============================================================================

# 使用 NVIDIA CUDA 基础镜像（cuRobo 需要 CUDA 11.8+）
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# CUDA 相关环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# 1. 安装系统依赖
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    # Python
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # OpenGL 和图形库（可视化需要）
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    # X11（可选，用于显示窗口）
    xvfb \
    x11-utils \
    # 其他库
    libeigen3-dev \
    libboost-all-dev \
    libassimp-dev \
    libfcl-dev \
    liboctomap-dev \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置 Python 3.10 为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# 2. 安装 Miniconda（用于安装 Pinocchio）
# =============================================================================
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda update -n base -c defaults conda -y \
    && conda clean -afy

# 配置 conda channels
RUN conda config --add channels conda-forge \
    && conda config --set channel_priority strict

# =============================================================================
# 3. 安装 Pinocchio（通过 Conda，最稳定的方式）
# =============================================================================
RUN conda install -y -c conda-forge \
    pinocchio \
    hpp-fcl \
    && conda clean -afy

# =============================================================================
# 4. 安装 PyTorch（支持 CUDA 11.8）
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

# =============================================================================
# 5. 安装 cuRobo（GPU 加速 IK 求解器）
# =============================================================================
# cuRobo 需要从源码安装以获得最佳兼容性
RUN pip install --no-cache-dir \
    "curobo[all]>=0.7.0" \
    || pip install --no-cache-dir curobo

# =============================================================================
# 6. 安装核心 Python 依赖
# =============================================================================
RUN pip install --no-cache-dir \
    # 科学计算
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    pandas>=2.0.0 \
    # URDF 解析
    urdf-parser-py>=0.0.4 \
    yourdfpy>=0.0.53 \
    # 可视化
    plotly>=5.15.0 \
    matplotlib>=3.7.0 \
    trimesh>=4.0.0 \
    networkx>=3.0 \
    # 3D 网页可视化
    meshcat>=0.3.2 \
    # 配置文件
    pyyaml>=6.0 \
    # 图像处理
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8 \
    Pillow>=10.0.0 \
    # Jupyter（可选，用于交互式开发）
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    # 物理仿真（可选）
    mujoco>=3.0.0

# =============================================================================
# 7. 设置工作目录和项目
# =============================================================================
WORKDIR /workspace

# 复制项目文件
COPY . /workspace/

# 安装项目依赖（如果有 requirements.txt）
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# =============================================================================
# 8. 环境变量配置
# =============================================================================
# OpenGL 渲染（无头模式）
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Jupyter 配置
ENV JUPYTER_ALLOW_INSECURE_WRITES=true

# cuRobo 配置
ENV CUROBO_DATA_DIR=/workspace/curobo_data

# Python 路径
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# =============================================================================
# 9. 暴露端口
# =============================================================================
# 8888: Jupyter Lab
# 7000: Meshcat (3D 网页可视化)
# 8050: Plotly Dash（可选）
EXPOSE 8888 7000 8050

# =============================================================================
# 10. 健康检查
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# =============================================================================
# 11. 入口点
# =============================================================================
# 默认启动 Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# 使用官方的 Miniconda3 镜像，省去安装 Conda 的麻烦
FROM continuumio/miniconda3

# 设置非交互前端
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装必要的系统图形库 (MuJoCo/Meshcat 需要)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# 2. 创建 Conda 环境并安装核心库
# 技巧：Pinocchio 在 conda-forge 源里是最稳定的
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda install -y python=3.10 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    jupyterlab \
    ipywidgets \
    # 核心机器人库
    pinocchio \
    meshcat \
    # 物理引擎
    mujoco \
    # 几何处理
    trimesh \
    networkx \
    && conda clean -ya

# 3. 补充 pip 安装 (Conda里没有的库)
# plotly 也可以用 conda 装，但 pip 有时更新
RUN pip install --no-cache-dir \
    plotly \
    urdf_parser_py \
    yourdfpy \
    imageio \
    imageio-ffmpeg

# 4. 设置环境变量
ENV MUJOCO_GL=osmesa
# 解决 Jupyter 权限问题
ENV JUPYTER_ALLOW_INSECURE_WRITES=true

# 5. 设置工作目录
WORKDIR /workspace

# 6. 暴露端口
# 8888: Jupyter Lab
# 7000: Meshcat (3D 网页可视化)
EXPOSE 8888 7000

# 7. 默认启动 Jupyter Lab，允许 root 运行，不打开浏览器，监听所有 IP
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

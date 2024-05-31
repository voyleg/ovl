FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Required for nvdiffrast and open3d
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgomp1 \
    cmake \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

# We install some packages from github further
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

# These are also required for nvdiffrast
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Setup python env
COPY requirements.txt /tmp
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Fix cuda-related bugs in pytorch image
# see https://github.com/microsoft/DeepSpeed/issues/2684#issuecomment-1379666816
ENV PATH /usr/local/cuda/bin:$PATH

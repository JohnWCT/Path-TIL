# ────────────────────────────────────────────────────────────────
#  TensorFlow 22.11 (TF-2, Python 3) – GPU 版基底
# ────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ────────────────────────────────────────────────────────────────
#  1. 系統層面相依套件
#     - libopenslide-dev：OpenSlide C 函式庫（TILscout WSI）
#     - libtiff5-dev / libjpeg-dev / libpng-dev：影像解碼
#     - libglib2.0-0 等：OpenCV runtime 依賴
# ────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenslide-dev \
        libtiff5-dev \
        libjpeg-dev \
        libpng-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────
#  2. Python 套件（版本鎖定）
#     * TensorFlow 2.10.0 已隨基底映像檔安裝，不必重裝 *
#
#     InceptionResNetV2_QuPath_2stage.py --aug 對照：
#       none / light / medium → tf.keras RandomFlip 等（無需額外套件）
#       heavy                 → imgaug + scikit-image（HED 色彩增強）
#     TILscout_edit.py        → tifffile（patch 寫入）
# ────────────────────────────────────────────────────────────────
RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
        'packaging>=24.0'

RUN python -m pip install \
        openslide-python==1.2.0 \
        opencv-python-headless==4.7.0.72 \
        scikit-learn==1.2.1 \
        pandas==1.4.4 \
        matplotlib==3.7.0 \
        numpy==1.23.5 \
        tifffile==2023.4.12 \
        imgaug==0.4.0 \
        scikit-image==0.19.3

WORKDIR /workspace

# 互動式終端機（例：docker run -it --gpus all -v "$(pwd)":/workspace path-til）
CMD ["/bin/bash"]

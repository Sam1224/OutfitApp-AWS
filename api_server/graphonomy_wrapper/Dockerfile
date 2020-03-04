#-----------------------------
# Docker イメージのベースイメージ
#-----------------------------
# CUDA 10.0 for Ubuntu 16.04
FROM nvidia/cuda:10.0-base-ubuntu16.04

#-----------------------------
# 基本ライブラリのインストール
#-----------------------------
# インストール時のキー入力待ちをなくす環境変数
ENV DEBIAN_FRONTEND noninteractive

RUN set -x && apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libx11-6 \
    python3-pip \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#-----------------------------
# 環境変数
#-----------------------------
ENV LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8
ENV PYTHONIOENCODING utf-8

ARG USER_NAME=ubuntu
ARG USER_ID=1000
ARG GROUP_NAME=sudo
ENV HOME=/home/${USER_NAME}
ARG WORK_DIR=${HOME}/share/graphonomy_wrapper

#-----------------------------
# ユーザーの追加
#-----------------------------
RUN useradd --create-home --uid ${USER_ID} --groups ${GROUP_NAME} --home-dir ${HOME} --shell /bin/bash ${USER_NAME}

# ディレクトリの所有権を設定： sudo グループの設定
RUN chown -R ${USER_NAME}:${GROUP_NAME} ${HOME}
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# user に sudo 権限を付与
RUN sudo gpasswd -a ${USER_NAME} sudo

# ディレクトリのアクセス権を設定：全アクセスを許可
RUN chmod 777 ${HOME}

# 指定ユーザーに切り替え
USER ${USER_NAME}

#-----------------------------
# 追加ライブラリのインストール
#-----------------------------
# miniconda のインストール
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=${HOME}/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# conda 上で Python 3.6 環境を構築
RUN ${HOME}/miniconda/bin/conda create -y --name py36 python=3.6.9 \
    && ${HOME}/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=${HOME}/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN ${HOME}/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
    && ${HOME}/miniconda/bin/conda clean -ya

# pytorch のインストール（CUDA 10.0-specific steps）
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py36_cu100" \
    && conda clean -ya

# OpenCV3 のインストール
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    && sudo rm -rf /var/lib/apt/lists/*

RUN conda install -y -c menpo opencv3=3.1.0 && conda clean -ya

# Apex (AMP) のインストール
RUN conda install -c conda-forge nvidia-apex && conda clean -ya
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex
#RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Others
RUN conda install -y tqdm && conda clean -ya
RUN conda install -y -c anaconda pillow==6.2.1 && conda clean -ya
RUN conda install -y -c anaconda networkx && conda clean -ya
RUN conda install -c anaconda scipy && conda clean -ya

# Other (for server)
RUN conda install -c anaconda flask && conda clean -ya
RUN conda install -c anaconda flask-cors && conda clean -ya
RUN conda install -c anaconda requests && conda clean -ya

#-----------------------------
# コンテナ起動後に自動的に実行するコマンド
#-----------------------------
#CMD ["/bin/bash"]

#-----------------------------
# コンテナ起動後の作業ディレクトリ
#-----------------------------
WORKDIR ${WORK_DIR}

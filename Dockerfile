From pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt upgrade -y --fix-missing && apt update --fix-missing && \
    apt install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \ # cv2
    wget \
    curl \
    git \
    zip \
    unzip \
    nkf \
    gcc \
    make \
    sudo \
    vim \
    silversearcher-ag \
    jq \
    tree \
    python3-dev python3-pip python3-setuptools \
    language-pack-ja-base \
    language-pack-ja \
        && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE="ja_JP:ja" && \
    apt autoremove && \
    apt clean

ENV LANG ja_JP.UTF-8

RUN pip install -U jupyterlab scikit-image albumentations
RUN mkdir /root/.jupyter
COPY jupyter_lab_config.py /root/.jupyter/
ENV PASSWORD password

WORKDIR /root

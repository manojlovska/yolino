# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #

##### WITHOUT CONDA ENVIRONMENT #####

# FROM python:3.8-bullseye
# # RUN apt update \
# #     && apt install git ffmpeg libsm6 libxext6 -y
# RUN python -m pip install --upgrade pip
# # RUN pip install virtualenv

# COPY . /usr/YOLinO/yolino/
# WORKDIR /usr/YOLinO/yolino/

# RUN pip install -e /usr/YOLinO/yolino/
# RUN export GIT_PYTHON_REFRESH=quiet
# RUN export DATASET_TUSIMPLE="/usr/YOLinO/tus_po_8p_dn19/TUSimple"
# RUN export PYTHONPATH=""


##### WITH CONDA ENVIRONMENT #####

FROM continuumio/miniconda3:latest

RUN apt update \
    && apt install git ffmpeg libsm6 libxext6 -y
RUN python -m pip install --upgrade pip

RUN conda create --name env_yolino python=3.8 --yes

ENV PATH /opt/conda/bin:$PATH

RUN /bin/bash -c "source activate env_yolino"

COPY . /usr/YOLinO/yolino/
WORKDIR /usr/YOLinO/yolino/

RUN pip install -e .
# RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

RUN echo "Make sure torch is installed:"
RUN python -c "import torch"

ENV GIT_PYTHON_REFRESH=quiet
ENV DATASET_TUSIMPLE="/usr/YOLinO/tus_po_8p_dn19/TUSimple"
ENV PYTHONPATH=""
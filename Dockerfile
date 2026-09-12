FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    SAM2_BUILD_CUDA=0 \
    SAM2_SIZE=large \
    SAM2_CKPT_DIR=/models \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends git wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Official SAM 2.1 (skip compiling the optional CUDA extension)
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2 \
    && SAM2_BUILD_CUDA=0 pip install --no-cache-dir -e /opt/sam2

# Bake SAM 2.1 Large so the first request is not a 900 MB download
RUN mkdir -p /models \
    && wget -q -O /models/sam2.1_hiera_large.pt \
       https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]

FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.2 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt $PROJECT_ROOT/

RUN pip3 install --no-cache-dir setuptools==41.0.0 && \
    pip3 install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]

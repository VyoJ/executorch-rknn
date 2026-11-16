# Ahead-of-time image: export PyTorch models and convert them with RKNN.
#
# Two Python environments, because rknn-toolkit2 1.6.0 pins old onnx/protobuf
# that cannot coexist with ExecuTorch:
#   /opt/venv   ExecuTorch, torch (default on PATH)
#   /opt/rknn   rknn-toolkit2 and only the dependencies its ONNX path needs
# Python 3.11 is the newest the toolkit supports and the oldest ExecuTorch 1.6 supports.
FROM --platform=linux/amd64 debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc

ARG TORCH_VERSION=2.15.0
ARG TORCHVISION_VERSION=0.29.2
ARG TORCHAO_VERSION=0.18.0
ARG EXECUTORCH_VERSION=1.6.0
ARG RKNN_TOOLKIT2_COMMIT=b25dadacc24b88eb7dfcaa47c9c525ecca89b319

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Model weights and calibration/evaluation images.
ENV TORCH_HOME=/opt/data/torch
RUN mkdir -p /opt/data/torch/hub/checkpoints && cd /opt/data \
    && curl -fsSL -o torch/hub/checkpoints/mobilenet_v2-b0353104.pth \
        https://download.pytorch.org/models/mobilenet_v2-b0353104.pth \
    && curl -fsSL -o imagenette2-160.tgz \
        https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz \
    && printf '%s  %s\n' \
        b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889 torch/hub/checkpoints/mobilenet_v2-b0353104.pth \
        64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5 imagenette2-160.tgz \
        | sha256sum -c - \
    && tar xzf imagenette2-160.tgz && rm imagenette2-160.tgz \
    && chmod -R a+rX /opt/data

# torch and torchao first from the CPU index so executorch's requirements are
# already satisfied and pip does not pull CUDA builds.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        "torchao==${TORCHAO_VERSION}" \
    && /opt/venv/bin/pip install --no-cache-dir \
        "executorch==${EXECUTORCH_VERSION}" numpy pytest onnx==1.23.2 onnxscript==0.7.2

# The wheel declares torch==1.13.1 and tensorflow==2.14.0, which have no cp311
# builds and are only used by load_pytorch/load_tensorflow, so install it
# without deps and add the rest of its requirements file by hand.
RUN python3 -m venv /opt/rknn \
    && curl -fsSL -o /tmp/rknn_toolkit2-1.6.0+81f21f4d-cp311-cp311-linux_x86_64.whl \
        "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/${RKNN_TOOLKIT2_COMMIT}/rknn-toolkit2/packages/rknn_toolkit2-1.6.0%2B81f21f4d-cp311-cp311-linux_x86_64.whl" \
    && echo "f1542483bcb53e99e5bc91d58229b4576046cf3e213522763ad81ef5e202d3d6  /tmp/rknn_toolkit2-1.6.0+81f21f4d-cp311-cp311-linux_x86_64.whl" | sha256sum -c - \
    && /opt/rknn/bin/pip install --no-cache-dir --no-deps /tmp/rknn_toolkit2-*.whl \
    && /opt/rknn/bin/pip install --no-cache-dir \
        onnx==1.14.1 onnxoptimizer==0.3.8 onnxruntime==1.16.0 protobuf==3.20.3 \
        "numpy<2" "psutil>=5.9.0" "ruamel.yaml>=0.17.21" "scipy>=1.9.3" \
        "tqdm>=4.64.1" "opencv-python-headless>=4.5.5.64" "fast-histogram>=0.11" \
        "Pillow>=10.0.1" \
    && rm /tmp/rknn_toolkit2-*.whl

ENV PATH="/opt/venv/bin:${PATH}"

ARG UID=1000
ARG GID=1000
RUN groupadd -o -g "${GID}" dev && useradd -o -m -u "${UID}" -g "${GID}" dev
USER dev
WORKDIR /work

# Ahead-of-time image: export PyTorch models to .pte (and later, convert with RKNN).
# Python 3.11: newest version rknn-toolkit2 1.6.0 supports, oldest ExecuTorch 1.6 supports.
FROM --platform=linux/amd64 debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc

ARG TORCH_VERSION=2.15.0
ARG TORCHVISION_VERSION=0.29.2
ARG TORCHAO_VERSION=0.18.0
ARG EXECUTORCH_VERSION=1.6.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# torch and torchao first from the CPU index so executorch's requirements are
# already satisfied and pip does not pull CUDA builds.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        "torchao==${TORCHAO_VERSION}" \
    && /opt/venv/bin/pip install --no-cache-dir \
        "executorch==${EXECUTORCH_VERSION}" numpy pytest

ENV PATH="/opt/venv/bin:${PATH}"

ARG UID=1000
ARG GID=1000
RUN groupadd -o -g "${GID}" dev && useradd -o -m -u "${UID}" -g "${GID}" dev
USER dev
WORKDIR /work

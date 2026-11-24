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

# Transformer models (spike D). transformers and optimum-executorch run only at
# export time. optimum-executorch 1.1.0 needs the transformers it pins (its
# static cache breaks on later versions); its other pins are skipped with
# --no-deps and it works with the versions above.
ARG SMOLLM2_REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
ARG WHISPER_REV=169d4a4341b33bc18d8881c4b69c2e104e1cc0af
ARG OPENAI_WHISPER_COMMIT=86098128c0b4f24f0e2aa2994de830614b474227
RUN /opt/venv/bin/pip install --no-cache-dir transformers==5.0.0rc1 soundfile==0.14.0 \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps \
        optimum-executorch==1.1.0 optimum==2.0.0 accelerate==1.16.0
RUN set -e; mkdir -p /opt/data/hf/SmolLM2-135M /opt/data/hf/whisper-tiny; cd /opt/data/hf \
    && hf=https://huggingface.co \
    && for f in config.json tokenizer.json model.safetensors; do \
         curl -fsSL -o SmolLM2-135M/$f $hf/HuggingFaceTB/SmolLM2-135M/resolve/${SMOLLM2_REV}/$f; done \
    && for f in config.json preprocessor_config.json tokenizer.json model.safetensors; do \
         curl -fsSL -o whisper-tiny/$f $hf/openai/whisper-tiny/resolve/${WHISPER_REV}/$f; done \
    && curl -fsSL -o jfk.flac https://raw.githubusercontent.com/openai/whisper/${OPENAI_WHISPER_COMMIT}/tests/jfk.flac \
    && printf '%s  %s\n' \
        1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843 SmolLM2-135M/config.json \
        9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c SmolLM2-135M/tokenizer.json \
        80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1 SmolLM2-135M/model.safetensors \
        ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98 whisper-tiny/config.json \
        9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666 whisper-tiny/preprocessor_config.json \
        7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395 whisper-tiny/model.safetensors \
        63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715 jfk.flac \
        | sha256sum -c - \
    && chmod -R a+rX /opt/data/hf

# MobileViT-XS for demo D1. Apple's research-only model licence; fetched here, never redistributed.
ARG MOBILEVIT_REV=0f4f298e136829c566224228d82066548a9082de
RUN set -e; mkdir -p /opt/data/hf/mobilevit-x-small; cd /opt/data/hf/mobilevit-x-small \
    && for f in config.json preprocessor_config.json pytorch_model.bin; do \
         curl -fsSL -o $f https://huggingface.co/apple/mobilevit-x-small/resolve/${MOBILEVIT_REV}/$f; done \
    && printf '%s  %s\n' \
        37a304f61c08ac7c485a21bb6000e24169de68b467c973a43b3e26e94082cfe0 config.json \
        ac69eb9624defabf01056e9dee3c68eb9d5f528a550007033c7574d7179249ec preprocessor_config.json \
        09b264fe681cb67144c41fe10b923c8d59ad4273f9d4cc3053080e6e986e63e8 pytorch_model.bin \
        | sha256sum -c - \
    && chmod -R a+rX /opt/data/hf

# The toolkit imports torch for some graph rewrites (LayerNorm folding); without
# it, plain-RKNN builds of transformer models stop with ModuleNotFoundError.
RUN /opt/rknn/bin/pip install --no-cache-dir torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

ENV PATH="/opt/venv/bin:${PATH}"

ARG UID=1000
ARG GID=1000
RUN groupadd -o -g "${GID}" dev && useradd -o -m -u "${UID}" -g "${GID}" dev
USER dev
WORKDIR /work

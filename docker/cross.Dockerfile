# Cross-compile image for Luckfox Pico (RV1103 / RV1106), armv7 + uClibc.
# Versions are pinned in docker/VERSIONS.md; keep the two in sync.
FROM --platform=linux/amd64 debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc

ARG LUCKFOX_SDK_COMMIT=824b817f889c2cbff1d48fcdb18ab494a68f69d1
ARG TOOLCHAIN=arm-rockchip830-linux-uclibcgnueabihf
ARG TORCH_VERSION=2.15.0

# g++ builds host tools (flatc) during the ExecuTorch build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git cmake ninja-build make file g++ \
        python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Kernel registration codegen imports torchgen, which ships inside the torch wheel.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==${TORCH_VERSION}" pyyaml typing_extensions

# Fetch only the toolchain directory from the (very large) Luckfox SDK repo.
RUN git clone --filter=blob:none --no-checkout https://github.com/LuckfoxTECH/luckfox-pico.git /tmp/sdk \
    && cd /tmp/sdk \
    && git sparse-checkout set --no-cone "/tools/linux/toolchain/${TOOLCHAIN}/" \
    && git checkout "${LUCKFOX_SDK_COMMIT}" \
    && mv "tools/linux/toolchain/${TOOLCHAIN}" /opt/toolchain \
    && rm -rf /tmp/sdk \
    && /opt/toolchain/bin/${TOOLCHAIN}-gcc --version

ENV PATH="/opt/venv/bin:/opt/toolchain/bin:${PATH}" \
    LUCKFOX_TOOLCHAIN=/opt/toolchain \
    LUCKFOX_TRIPLE=${TOOLCHAIN}

# Non-root user; UID/GID passed from the host by scripts/dk.
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g "${GID}" dev && useradd -o -m -u "${UID}" -g "${GID}" dev
USER dev
WORKDIR /work

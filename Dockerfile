# Ollama with CUDA for the Jetson Nano (Tegra X1, sm_53) on Pantavisor.
#
# The Nano is held at L4T R32 / CUDA 10.2, whose nvcc accepts GCC 8 at most;
# current Ollama needs CUDA 11 and GCC 11. Ollama b9f74ff (v0.1.33 era,
# ollama/ollama#10090) is the last that builds there, against the CUDA 10.2
# toolkit and GCC 8, running its llama.cpp runner on the GPU.
#
# The device's nvidia driver only loads nvgpu: the image carries libcuda.so.1
# and the Tegra libraries it needs, from the L4T release the BSP's nvgpu comes
# from (R32.7.6, t210). CUDA runtime and cuBLAS travel inside the ollama
# binary, as Ollama ships them.

ARG L4T_VERSION=32.7.6-20241104234540

FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS l4t
ARG L4T_VERSION
ARG L4T=https://repo.download.nvidia.com/jetson/t210/pool/main/n
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl binutils xz-utils zstd \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /l4t
RUN set -e; \
    for p in "nvidia-l4t-core 61ada8308bd38fe6fe1793b658c83c3ef3dcc057a0d9e080dc0336e61be683c2" \
             "nvidia-l4t-cuda 0042650294496d632f5d7695a23632411ec3c1d3b2fdd664181338031bb45f86" \
             "nvidia-l4t-3d-core 44df4b465417916ac1bdb1c801892c7af948114cadf5a3a1eafb780984600a94"; do \
      set -- $p; deb="$1_${L4T_VERSION}_arm64.deb"; \
      curl -fsSL -o "$deb" "$L4T/$1/$deb"; echo "$2  $deb" | sha256sum -c -; \
      mkdir -p x && (cd x && ar x "../$deb" && tar -xf data.tar.*); rm -f x/*.tar.* x/debian-binary; \
    done; \
    mkdir -p /out/usr/lib/aarch64-linux-gnu/tegra; \
    cd x/usr/lib/aarch64-linux-gnu/tegra; \
    cp -a libcuda.so.1.1 libnvrm_gpu.so libnvrm.so libnvrm_graphics.so libnvos.so \
          libnvidia-fatbinaryloader.so.440.18 libnvidia-ptxjitcompiler.so.440.18 \
          /out/usr/lib/aarch64-linux-gnu/tegra/; \
    cd /out/usr/lib/aarch64-linux-gnu/tegra; \
    ln -s libcuda.so.1.1 libcuda.so.1; ln -s libcuda.so.1 libcuda.so; \
    ln -s libnvidia-ptxjitcompiler.so.440.18 libnvidia-ptxjitcompiler.so.1; \
    mkdir -p /out/etc/ld.so.conf.d; \
    echo /usr/lib/aarch64-linux-gnu/tegra > /out/etc/ld.so.conf.d/nvidia-tegra.conf

# Built natively for arm64 (emulated on other hosts): Ollama's generate step
# does not cross-compile, and this is the toolchain ollama/ollama#10090 used.
FROM ubuntu:18.04 AS build
ARG OLLAMA_COMMIT=b9f74ff3d643fd2b579bfc24132e34f3bc7bf777
ARG LLAMA_CPP_COMMIT=952d03dbead16e4dbdd1d3458486340673cc2465
ARG GO_VERSION=1.22.12
ARG CMAKE_VERSION=3.27.9
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git build-essential gcc-8 g++-8 \
 && rm -rf /var/lib/apt/lists/*
# CUDA 10.2 for Jetson (JetPack 4.6, L4T R32.7).
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | apt-key add - \
 && echo "deb https://repo.download.nvidia.com/jetson/common r32.7 main" > /etc/apt/sources.list.d/nvidia-l4t.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      cuda-nvcc-10-2 cuda-cudart-dev-10-2 cuda-driver-dev-10-2 libcublas-dev libcublas10 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sfn /usr/local/cuda-10.2 /usr/local/cuda \
 && test -x /usr/local/cuda/bin/nvcc && ls /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/stubs/libcuda.so
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz \
      | tar -xz -C /opt && ln -s /opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/cmake /usr/local/bin/cmake \
 && curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-arm64.tar.gz | tar -xz -C /usr/local
ENV PATH=/usr/local/go/bin:/usr/local/cuda/bin:$PATH \
    CC=gcc-8 CXX=g++-8 CUDAHOSTCXX=g++-8 \
    GOTOOLCHAIN=local GOFLAGS=-buildvcs=false
WORKDIR /src
RUN git init -q ollama && cd ollama \
 && git remote add origin https://github.com/ollama/ollama.git \
 && git fetch -q --depth 1 origin ${OLLAMA_COMMIT} && git checkout -q FETCH_HEAD \
 && git init -q llm/llama.cpp && cd llm/llama.cpp \
 && git remote add origin https://github.com/ggerganov/llama.cpp.git \
 && git fetch -q --depth 1 origin ${LLAMA_CPP_COMMIT} && git checkout -q FETCH_HEAD
WORKDIR /src/ollama
# llama.cpp patches, applied by Ollama's generate step with its own.
COPY patches/*.diff llm/patches/
# sm_53 only, C++11 for CUDA 10.2's nvcc; libcuda.so for linking is the
# toolkit's stub (the real one is the device's, in the runtime image).
ENV CMAKE_CUDA_ARCHITECTURES=53 \
    OLLAMA_CUSTOM_CUDA_DEFS="-DCMAKE_CUDA_STANDARD=11 -DCMAKE_CUDA_HOST_COMPILER=g++-8 -DLLAMA_CUDA_FORCE_MMQ=on" \
    CUDA_LIB_DIR=/usr/local/cuda/lib64 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs
# Fail here rather than ship a CPU-only engine.
RUN go generate ./... && ls llm/build/linux/arm64/cuda_v10/bin/ \
 && go build -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=0.1.33-jetson" -o /out/ollama .

FROM ubuntu:18.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=l4t /out/ /
COPY --from=build /out/ollama /usr/bin/ollama
RUN ldconfig
# The container keeps the host network namespace (the docker template's
# default for Pantavisor apps): this is the device's port on the LAN, open to
# the Hollama UI served from the device's port 4173.
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_ORIGINS=* \
    OLLAMA_MODELS=/models \
    OLLAMA_KEEP_ALIVE=10m \
    HOME=/models
RUN mkdir -p /models
VOLUME ["/models"]
EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]

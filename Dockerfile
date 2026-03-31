# NeuroSwarm Base Image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV PATH=/usr/local/cuda/bin:${PATH}

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libspdlog-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT-LLM (simulated for base image)
# Run pip install tensorrt-llm==0.10.0 --extra-index-url https://pypi.nvidia.com

# Clone and build Prometheus CPP client
WORKDIR /tmp/build
RUN git clone -b v1.1.0 --recursive https://github.com/jupp0r/prometheus-cpp.git && \
    cd prometheus-cpp && \
    cmake -B build -DBUILD_SHARED_LIBS=ON -DENABLE_PUSH=OFF -DENABLE_COMPRESSION=OFF && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    rm -rf /tmp/build/prometheus-cpp

# Build NeuroSwarm
WORKDIR /workspace
COPY . /workspace

RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" && \
    cmake --build . --parallel $(nproc)

# Install Python package
RUN pip3 install -e ./python[all]

# Setup for testing
RUN mkdir -p /tmp/neuroswarm/logs \
    /tmp/neuroswarm/profiles \
    /tmp/neuroswarm/bugs \
    /tmp/neuroswarm/reports

# Production Image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    nsight-systems-2024.2.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries and libraries
COPY --from=builder /workspace/build/neuroswarm /usr/local/bin/
COPY --from=builder /workspace/build/libneuroswarm.so /usr/local/lib/
COPY --from=builder /usr/local/lib/libprometheus-cpp* /usr/local/lib/
COPY --from=builder /workspace/python /workspace/python

WORKDIR /workspace/python
RUN pip3 install -e .[all]

# Default command
CMD ["neuroswarm", "--config", "/workspace/configs/default_config.json"]

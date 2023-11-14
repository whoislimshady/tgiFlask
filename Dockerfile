# Use the NVIDIA CUDA base image
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    unzip \
    pkg-config \
    libssl-dev \
    gcc \
    nano \
    python3 \
    python3-pip

# Clone the GitHub repository
RUN git clone https://github.com/huggingface/text-generation-inference.git


RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo --help
# 
# Change to the repository directory
WORKDIR /text-generation-inference


# Install Protocol Buffers
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

# Set environment variables
ENV HUGGING_FACE_HUB_TOKEN=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

# Create model directory
RUN mkdir /text-generation-inference/model
WORKDIR /text-generation-inference

RUN BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels

COPY . .
RUN pip  install --ignore-installed flask
RUN pip  install --ignore-installed schedule
RUN pip  install --ignore-installed pymongo
RUN cd ..




CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
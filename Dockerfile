FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg \
    sox libsox-dev libsox-fmt-all curl build-essential

# install rust and cargo (required for tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --no-cache-dir --upgrade pip

# install torch with cu124
RUN python3 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# install pinned dependencies
RUN python3 -m pip install --no-cache-dir \
    accelerate==1.10.1 \
    datasets==3.3.2 \
    evaluate==0.4.5 \
    huggingface-hub==0.34.4 \
    jiwer==4.0.0 \
    librosa==0.11.0 \
    transformers==4.50.0 \
    wandb==0.21.3 \
    hf_transfer

# install remaining pinned packages
RUN python3 -m pip install --no-cache-dir \
    aiohttp==3.12.15 \
    filelock==3.13.1 \
    fsspec==2024.6.1 \
    Jinja2==3.1.4 \
    numpy==2.1.2 \
    pandas==2.3.2 \
    pillow==11.0.0 \
    protobuf==6.32.0 \
    pyarrow==21.0.0 \
    pydantic==2.11.7 \
    PyYAML==6.0.2 \
    regex==2025.8.29 \
    requests==2.32.5 \
    safetensors==0.6.2 \
    scikit-learn==1.7.1 \
    scipy==1.15.3 \
    soundfile==0.13.1 \
    tokenizers==0.21.4 \
    tqdm==4.67.1

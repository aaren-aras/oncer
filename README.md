# Oncer
Hey! This web app lets you upload brain MRI scans and analyze them with a ResU-Net <u>c</u>onvolutional <u>n</u>eural <u>n</u>etwork (CNN). Specifically, the model performs **tumour segmentation** and subtype classification, trained on the BraTS 2021 glioma dataset across four MRI modalities: T1, T1CE, T2, and FLAIR.

[insert host link here]

[insert video demo here]

## Prerequisites
You'll need *three* main pieces in place before running this project locally:

### 1. OS & Environemnt
 - On Windows, install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Ubuntu **22.04** recommended)
 - On Linux, you can run Docker natively
 - On macOS, GPU acceleration isn’t supported (unless you’re using CPU-only builds)

### 2. Docker with NVIDIA GPU Support
 - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
 - Install the latest [NVIDIA drivers](https://www.nvidia.com/en-us/drivers/) for your GPU
 - Install the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** so Docker can use your GPU

Verify setup:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 3. Project Data (not included in this repo)
 - Download the brain MRI scans for training the CNN from Kaggle, [here](https://www.kaggle.com/datasets/praneet0327/brain-tumor-dataset/data)
 - Inside the `api` folder, create a new folder called `data`
 - Place the downloaded `BraTS2021_Training_Data` folder inside it, such that the final path is: `api/data/BraTS2021_Training_Data`

These steps ensure that the `preprocessing.py` and `model.py` scripts in `api/services/scripts` can automatically locate the dataset without additional configuration.

## Setup
```bash
# Clone repo and navigate to project dir
git clone https://github.com/aaren-aras/oncer.git && cd oncer
```

To avoid compatibility issues with the latest NVIDIA GPUs (and the ensuing CUDA/cuDNN mismatch headaches), I decided to use NVIDIA's <u>N</u>VIDIA <u>G</u>PU <u>C</u>LOUD (NGC) TensorFlow containers, which come pre-packaged with the correct versions of TensorFlow, CUDA, and cuDNN.

```bash
# Pull the latest container image
docker pull nvcr.io/nvidia/tensorflow:25.02-tf2-py3 

# Run the container interactively with GPU access and shared memory for OpenCV
docker run --gpus all -it --rm \ 
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \ 
    -v /path/to/Oncer:/workspace/Oncer \ 
    nvcr.io/nvidia/tensorflow:25.02-tf2-py3
```

Inside the container:

```bash
# Install any missing system libraries for OpenCV
apt update && apt install -y libgl1 libglib2.0-0

# Install Python dependencies
cd /workspace/Oncer/api
pip install -r requirements.txt

# Prepare BraTS 2021 data and generate model files
python -m src.services.scripts.data
python -m src.services.scripts.model

# Verify GPU is working
python -c "import tensorflow as tf; print('Available GPUs:', tf.config.list_physical_devices('GPU'))"
nvcc --version  
nvidia-smi
```

Alternatively, if you don't want to use Docker, you *can* choose to set the project up locally (Python/Node), but you'll need to **manually** install compatible CUDA + cuDNN versions (refer to this [table](https://www.tensorflow.org/install/source#gpu) for tested build configurations). 
 - TensorFlow GPU support outside containers is only guaranteed up to TF **2.10 (CUDA 11.2, cuDNN 8.1)**
 - If you have a newer GPU (e.g., RTX 40/50 series), containers are STRONGLY recommended

```bash
# ALTERNATIVELY: install Python deps within virtual env (Windows example)
cd api 
py -3.10 -m venv .venv
source .venv/Scripts/activate # Git Bash
pip install -r requirements.txt

# Prepare BraTS 2021 data and generate model files
python -m src.services.scripts.data
python -m src.services.scripts.model
```

Once the model's been trained, launch the web app locally with Node.js:

```bash
# Install Node.js deps
cd oncer && npm install

# Start dev server
npm run start

# Build and preview prod
npm run build
npm run preview
```

## Issues
 - TF GPU compatibility is fragile outside Docker
 - Windows requires WSL2 for Docker GPU acceleration

## Retrospective
 - Consider rewriting entire backend in Python 
 - Docker NGC containers MASSIVELY simplify GPU + CUDA setup for modern NVIDIA GPUs requiring CUDA 12+ (no version mismatch, no DLL errors, no local TF rebuilds, ...)
   - Don't waste time with local installs and juggling Python versions, CUDA toolkits, and cuDNN DLLs on Windows when there're cleaner solutions available

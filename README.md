# Oncer
Hey! This web app lets you perform **2D brain tumour segmentation** on **structural MRI scans** with a ResU-Net <u>c</u>onvolutional <u>n</u>eural <u>n</u>etwork (CNN). The model is trained on a dataset from the [2021 RSNA-ASNR-MICCAI BraTS Challenge](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/) across *4* MRI modalities: `T1`, `T1CE`, `T2`, and `FLAIR`.

[insert host link here]

[insert video demo here]

## Prerequisites
You'll need *3* main pieces in place before running this project locally:

### 1. OS & Environemnt
 - On Windows, install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Ubuntu **22.04** recommended)
 - On Linux, you can run Docker natively
 - On macOS, GPU acceleration isn’t supported (unless you’re using CPU-only builds)

### 2. Docker with NVIDIA GPU Support
 - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
 - Install the latest [NVIDIA drivers](https://www.nvidia.com/en-us/drivers/) for your GPU
 - Install the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** so that Docker can use your GPU

Verify setup:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 3. Project Data (not included in this repo)
 - Download the brain MRI scans from Kaggle, [here](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data)
 - Inside the `api` directory, create a new folder and name it `data`
 - Place the downloaded `BraTS2021_Training_Data` folder inside `data` (`api/data/BraTS2021_Training_Data`)

These steps ensure that the `preprocessing.py` and `model.py` scripts in `api/src/scripts` can find the dataset without additional configuration.

## Setup
```bash
# Clone repo and navigate to project dir
git clone https://github.com/aaren-aras/oncer.git && cd oncer
```

From the project **root**, create a `.env` file and set a port number for the backend (e.g., `5000`):

```
API_PORT=5000
```

Then, run the following:

```bash
chmod +x api/entrypoint.sh
```

### Option A: Docker / NGC (recommended)
To avoid compatibility issues with the latest NVIDIA GPUs (and the ensuing CUDA/cuDNN mismatch headaches), I decided to use NVIDIA's <u>N</u>VIDIA <u>G</u>PU <u>C</u>loud (NGC) [TensorFlow containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) for **GPU-accelerated** CNN training on Windows. These containers come pre-packaged with versions of TensorFlow, CUDA, and cuDNN that are (almost) **guaranteed** to work together, alongside other stuff for optimizing GPU performance.

After launching **Docker Desktop**:

```bash
# Build backend image
cd api && docker build -t oncer-api .

# Run backend locally (available @ https://localhost:${API_PORT})
cd api && docker compose up --build
```

You can close the container with `docker compose down`. 

### Option B: Local 
Alternatively, if you don't want to use Docker/NGC, you *could* set up Python, TensorFlow, [CUDA](https://developer.nvidia.com/cuda-toolkit-archive), and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) **manually** from your end. Of course, this means YOU are responsible for making sure ALL versions play nicely together: a fate I *personally* wouldn't wish on my worst enemy. But hey, the choice is yours! 

Refer to this [table](https://www.tensorflow.org/install/source#gpu) for tested build configurations.

#### Sidenotes
 - TensorFlow GPU support outside containers is only guaranteed up to **2.10 (CUDA 11.2, cuDNN 8.1)** for Windows
 - If you have a newer GPU (e.g., RTX 40/50 series), containers are STRONGLY recommended

```bash
cd api

# Install Python deps within virtual env (Windows example)
py -3.10 -m venv .venv
source .venv/Scripts/activate # Git Bash
pip install -r requirements.txt

# Run backend
./entrypoint.sh
```

### Running Locally
In another terminal:

```bash
# Install Node.js deps 
cd ../app && npm install

# Launch dev instance (frontend)
npm run start

# Build and preview prod instance
npm run build
npm run preview
```

## Issues
 - TF GPU compatibility is fragile outside Docker
 - Windows requires WSL2 for Docker GPU acceleration

## Retrospective
 - Please stop falling for **scope creep**
 - KNOW your **tools**, and when/how/why to use them given the scope
   - Consider writing the entire backend in Python for all future DL projects
 - Docker NGC containers MASSIVELY simplify GPU + CUDA setup for modern NVIDIA GPUs requiring CUDA 12+ (no version mismatch and local rebuild trial-and-error, no Bazel errors, no DLL errors, ...)
   - Don't waste time with local installs and juggling Python versions, CUDA toolkits, and cuDNN DLLs on Windows when there're cleaner solutions available
   - You don't have to use *older* versions of software to achieve compatibility
   - Some TF builds lack precompiled CUDA kernels for newer GPUs with higher compute capabilities, forcing them to JIT-compile PTX at runtime, which can drastically slow startup; these containers avoid the issue by including prebuilt, GPU-optimized binaries
 - Look into multi-GPU setups (distributed training?): https://developer.nvidia.com/nccl

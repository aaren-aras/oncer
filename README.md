# Oncer
This web app enables users to upload medical images and have them processed by a convolutional neural network (CNN) for tumour detection. 

[insert host link here]

[insert video demo here]

## Prerequisites
This project relies on specific versions of **Node.js** and **Python** to work properly due to weird compatibility issues with [TensorFlow.js](https://www.tensorflow.org/js). I've personally found the most success with Node.js v**20.17** and Python **3.8.10**, but feel free to experiment a bit!

It also requires `data` and `model` files that have not been provided in this repository. After downloading the brain MRI scans from [here](https://www.kaggle.com/datasets/praneet0327/brain-tumor-dataset/data), place all contents (both "Negative"- and "Positive"-labelled) inside the `api` folder, following the path `api/data/brain`. The Python files in `api/src/utils` will handle the rest (see setup instructions below).

## Issues
TBA

## Setup
```bash
# Clone repository
$ git clone https://github.com/aaren-aras/oncer.git

# Navigate into project directory
$ cd oncer

# Install Node.js dependencies
$ npm install

# Install Python dependencies
$ cd api && pip install virtual env
$ virtualenv -p C:/Users/[User]/AppData/Local/Programs/Python/Python38/python.exe .venv # default path
$ source .venv/Scripts/activate # for Windows
$ pip install -r requirements.txt

# Generate model files
$ cd api/src/utils && python model.util.py

# Start development server
$ npm run start

# Build and preview production files
$ npm run build
$ npm run preview

```

## Retrospective
 - Consider writing the entire backend in Python (with a framework like Flask) for more uniformity 
 - ... (TBA)
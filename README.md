# HKU-DASC7606-A2
HKU DASC 7606 Assignment 2 (Computer Vision: Image Generation), 2024-2025 Fall

This codebase is only for HKU DASC 7606 (2024-2025) course. Please don't upload your answers or this codebase to any public platforms (e.g., GitHub) before permitted. All rights reserved.

# Introduction
Diffusion models have emerged as powerful generative frameworks capable of producing high-quality samples from complex distributions. These models operate by gradually adding noise to data and then learning to reverse this diffusion process to recover the original data. In this assignment, you will explore the fundamentals of diffusion models, specifically the Denoising Diffusion Probabilistic Model (DDPM). You will apply these concepts by training a DDPM with U-Net architecture on the MNIST handwritten digits dataset. By the end of this assignment, you will gain hands-on experience with the training and implementation of diffusion models, as well as an understanding of their strengths and limitations in generating synthetic data.

# Environment Setup

Note: you may also reuse the environment from the previous assignment (cv_env) if you have already set it up. 


**Installing Python 3.8+**: 
To use python3, make sure to install version 3.8+ on your machine.

**Virtual environment**: The use of a virtual environment via Anaconda is recommended for this project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies are installed. To establish a Conda virtual environment, execute the following commands:
```bash
git clone https://github.com/Hans-Lan/DASC7606-A2.git
conda create -n ddpm python=3.10
conda activate ddpm
```
Follow the official PyTorch installation guidelines to set up the PyTorch environment. This guide uses PyTorch version 2.0.1 with CUDA 11.8. Alternate versions may be specified by adjusting the version number:
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```
Install other requirements:

```bash
pip install tqdm
```

# 3 Working on the assignment

## 3.1 Code & Data
We have provided you with the initial codebase, which is defined in three Python files:

- **model.py** - Contains the definition of the diffusion model.
- **unet.py** - Contains the definition of the U-Net model that handles the denoising process.
- **train_mnist.py** - The main training script. Run `python train_mnist.py` to start training and evaluating the model.

The provided code is not complete. What you need to do is fill in the missing code blocks according to the requirements to complete the training for unconditional image generation.

In this assignment, we use the MNIST handwritten digits dataset as the training data. The dataset will be automatically downloaded the first time you run the `train_mnist.py` with the `create_mnist_dataloaders` function. If the download fails, you can manually download it from [this link](https://drive.google.com/file/d/11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E/view) and extract it to the directory where the code is located. The final file structure will be as follows:

```
minist_data/MNIST/raw/
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte
- train-images-idx3-ubyte
- train-labels-idx1-ubyte
```

## 3.2 Assignment tasks
**Task 1: Unconditional generation (70%)**

In this task, you will explore the principles of the Denoising Diffusion Probabilistic Model (DDPM). You are required to complete five code blocks in the `model.py` and `unet.py` files to successfully build and train a model that generates handwritten digits from the MNIST dataset. Understanding the DDPM algorithm and U-Net architecture is crucial for this task. If you have problems in implementing the algorithm, we strongly recommend you to read [Lilian Weng's blog about DDPM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) and [the U-Net paper](https://arxiv.org/abs/1505.04597).

**Task 2: Conditional Generation (10%)**  
This task involves extending the existing code to implement conditional generation, that is, to enable the generation of images corresponding to specific digits. The implementation should be detailed in your final report.

*Hint*: You may refer to the handling of time embedding in Unet to incorporate digit labels as conditions during the reverse diffusion process.

**Task 3: Write a Report (20%, max 2 pages)**  
Your report should be structured into three main sections: Introduction, Method, and Experiment. It should highlight the effectiveness of the image generation, showcasing outputs for all digits from 0 to 9. Additionally, include an analysis of your experimental results, such as loss curves, the impact of the variance schedule, sampling steps, normalization, and the effects of the clipping operation during reverse diffusion. You are encouraged to explore further insights and analyses that arise from your experiments.

## 3.3 Files to submit
1. Final Report (PDF, up to 2 pages)

2. Codes

    You are required to submit your code for Task 1 within the given files. If you complete Task 2, please include a text file that explains how to run your extended implementation.

3. Model Weights

    If your checkpoint is less than 30MB, you can directly submit the model weights. 

    If your checkpoint is larger than 30MB, you can submit the model weights in the format of model checkpoint link (model_link.txt) due to the limitation on submission file size. Recommended to use Google Drive or Dropbox to share the model weights.

    Please ensure adherence to model naming conventions and ensure compatibility with the code.

If your student id is 30300xxxxx, then the compressed file for submission on Moodle should be organized as follows:
```
30300xxxxx.zip
├── report.pdf
├── your code
├── checkpoint / model_link.txt
└── (optional) README.md
```

## 3.4 Timeline

October 6, 2024 (Sun.): The assignment release.  
November 10, 2024 (Sun.): Submission deadline (23:59 GMT+8).

Late submission policy:

- 10% for late assignments submitted within 1 day late. 
- 20% for late assignments submitted within 2 days late.
- 50% for late assignments submitted within 7 days late.
- 100% for late assignments submitted after 7 days late.

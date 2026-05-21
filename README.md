# CUDA Practices with WSL

## Description

This repository contains a series of basic CUDA practices made in C/C++.

The goal is to learn the fundamentals of CUDA programming, GPU execution, threads, blocks, kernels, and GPU properties in a simple way.

These practices are focused on beginners who are starting with CUDA development.

## Important Note

These setup instructions are only for:

- Windows
- WSL2
- Ubuntu on WSL

## Practices Included

- GPU device properties
- Kernel launch
- Threads and blocks indexing
- CUDA data transfer
- 3D grids and thread blocks
- Basic CUDA execution examples

## WSL Environment Setup

### 1. Check WSL and GPU

Open PowerShell in Windows and run:

```powershell
wsl --status
wsl --update
```

Then open Ubuntu WSL and run:

```bash
nvidia-smi
```

If you can see your NVIDIA GPU information, then WSL can correctly detect your GPU.

### 2. Install Basic Development Tools

Inside Ubuntu WSL, run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential g++ -y
```

These packages install the basic tools needed to compile C/C++ programs.

### 3. Install CUDA Toolkit in WSL

The safest way is to use NVIDIA’s official CUDA Toolkit page.

Select:

- Operating System: Linux
- Architecture: x86_64
- Distribution: WSL-Ubuntu
- Version: Your Ubuntu version
- Installer Type: deb local or network

Then copy and run the commands NVIDIA provides.

CUDA versions change, so the official NVIDIA instructions should be followed.

Official CUDA Toolkit page:

https://developer.nvidia.com/cuda-downloads

Official CUDA WSL documentation:

https://docs.nvidia.com/cuda/wsl-user-guide/index.html

### 4. Verify CUDA Installation

After installing CUDA, run:

```bash
nvcc --version
```

If CUDA was installed correctly, you should see the CUDA compiler version.

## Compiling CUDA Programs

CUDA files usually use the `.cu` extension.

To compile a CUDA program:

```bash
nvcc file_name.cu -o output_name
```

To run it:

```bash
./output_name
```

## Example

Compile:

```bash
nvcc gpu_properties.cu -o gpu_properties
```

Run:

```bash
./gpu_properties
```

## Purpose of These Practices

These practices help understand:

- How CUDA interacts with the GPU
- How kernels are launched
- How threads and blocks work
- How to transfer data between CPU and GPU
- What the limits of the GPU are
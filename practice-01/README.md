# CUDA GPU Properties

## Description

This program uses CUDA and C++ to display basic information about the GPU installed on the computer.

The program creates a variable of type `cudaDeviceProp` and uses it to read the properties of the CUDA device. This is useful because it helps us understand the limits of the GPU before writing CUDA programs.

## Information shown

The program prints:

- GPU Name
- Multiprocessor count (MP)
- Max blocks per Multiprocessor
- Max threads per Multiprocessor
- Max threads per block
- Max grid size in X, Y, and Z
- Max threads per block dimension in X, Y, and Z
- Warp size

## Requirements

Run:

```bash
nvcc device_properties.cu -o device_properties
./device_properties
```

To run this program, you need:

- A computer with an NVIDIA GPU
- CUDA installed
- `nvcc` compiler working

[!NOTE]
Check instructions on installing CUDA on the main directory


You can check if `nvcc` is installed with:

```bash
nvcc --version
```
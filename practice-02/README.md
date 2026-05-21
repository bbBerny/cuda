# CUDA Data Transfer Practice

## Description

This program demonstrates how memory mapping, allocation, and data transfer work across different spaces in CUDA.

The objective is to move a 1D vector through multiple host memory locations and separate GPU devices, if available, or distinct memory blocks using standard CUDA memory copying routines with `cudaMemcpy`.

The data lifecycle follows a strict ring-like pipeline sequence:

```text
HostA → DeviceA
DeviceA → DeviceB
DeviceB → HostB
HostB → HostC
```

At the final step, the program prints the arrays at `HostA` and `HostC` to verify data integrity.

## CUDA Data Transfer Types

CUDA uses different internal transfer paths depending on the `cudaMemcpyKind` flag used in `cudaMemcpy`.

### `cudaMemcpyHostToDevice`

Copies memory from the host system RAM to the global memory of the GPU.

This represents a CPU to GPU transfer.

### `cudaMemcpyDeviceToDevice`

Transfers data from one location in GPU memory to another.

This can happen inside the same GPU or between GPU devices if peer-to-peer transfer is available.

### `cudaMemcpyDeviceToHost`

Copies data from GPU global memory back to the host system RAM.

This represents a GPU to CPU transfer.

### `cudaMemcpyHostToHost`

Copies data between two host memory locations.

This is similar to a normal CPU-side `memcpy`, but executed through the CUDA runtime.

## Requirements

To run this practice exercise, you need:

- An NVIDIA GPU with CUDA support
- CUDA libraries installed
- The `nvcc` compiler installed and configured in your shell path

## Compile

Compile the implementation file with:

```bash
nvcc data_transfer.cu -o data_transfer
```

## Run

Run the compiled executable with:

```bash
./data_transfer
```

## Expected Result

The program should print the original array from `HostA` and the final copied array from `HostC`.

If both arrays show the same values, the data transfer pipeline worked correctly.

## Purpose

This practice helps understand how CUDA moves data between CPU memory and GPU memory.

It also shows why memory transfers are important in CUDA programming, since GPU programs often need to move data before and after kernel execution.
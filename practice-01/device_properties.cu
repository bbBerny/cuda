#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main() {
    // 1. Initialize to 0 so we don't read garbage
    int deviceCount = 0; 
    
    // 2. Catch the actual error returned by CUDA
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    if (deviceCount == 0) {
        cout << "No CUDA GPU found." << endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        cout << "Failed to get device properties: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // If it gets here, the data is 100% real and safe to print!
    cout << "GPU Name: " << prop.name << endl;
    cout << "Multiprocessor count (MP): " << prop.multiProcessorCount << endl;
    cout << "Max blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << endl;
    cout << "Max threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Warp size: " << prop.warpSize << endl;

    return 0;
}
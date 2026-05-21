#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main() {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        cout << "No CUDA GPU found." << endl;
        return 1;
    }

    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    cout << "GPU Name: " << prop.name << endl;
    cout << "Multiprocessor count (MP): " << prop.multiProcessorCount << endl;
    cout << "Max blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << endl;
    cout << "Max threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;

    cout << "Max grid size:" << endl;
    cout << "  X: " << prop.maxGridSize[0] << endl;
    cout << "  Y: " << prop.maxGridSize[1] << endl;
    cout << "  Z: " << prop.maxGridSize[2] << endl;

    cout << "Max threads per block dimension:" << endl;
    cout << "  X: " << prop.maxThreadsDim[0] << endl;
    cout << "  Y: " << prop.maxThreadsDim[1] << endl;
    cout << "  Z: " << prop.maxThreadsDim[2] << endl;

    cout << "Warp size: " << prop.warpSize << endl;

    return 0;
}
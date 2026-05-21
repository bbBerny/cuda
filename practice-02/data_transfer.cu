#include <iostream>
#include <cuda_runtime.h>

#define VECTOR_SIZE 8

// Helper macro for error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    const size_t bytes = VECTOR_SIZE * sizeof(int);

    // --- 1. Host Memory Allocation ---
    int* hostA = (int*)malloc(bytes);
    int* hostB = (int*)malloc(bytes);
    int* hostC = (int*)malloc(bytes);

    // Initialize HostA with arbitrary values
    std::cout << "Initializing HostA: ";
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        hostA[i] = (i + 1) * 11; // Arbitrary distinct numbers
        std::cout << hostA[i] << " ";
        hostB[i] = 0;            // Zero out intermediate buffers
        hostC[i] = 0;
    }
    std::cout << "\n\n";

    // --- 2. Determine Device Configuration ---
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    int devA_ID = 0;
    int devB_ID = 0;

    // Multi-GPU detection fallback logic
    if (deviceCount >= 2) {
        devB_ID = 1;
        std::cout << "System has multiple GPUs. Using Device 0 (DeviceA) and Device 1 (DeviceB).\n";
    } else {
        std::cout << "Single GPU detected. Virtualizing separate DeviceA and DeviceB regions on Device 0.\n";
    }

    // --- 3. Device Memory Allocation ---
    int *deviceA = nullptr;
    int *deviceB = nullptr;

    CUDA_CHECK(cudaSetDevice(devA_ID));
    CUDA_CHECK(cudaMalloc((void**)&deviceA, bytes));

    CUDA_CHECK(cudaSetDevice(devB_ID));
    CUDA_CHECK(cudaMalloc((void**)&deviceB, bytes));

    std::cout << "Executing Data Transfer Pipeline...\n";

    // --- 4. Pipeline Execution Sequencer ---

    // Sequence 1: HostA -> DeviceA
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice));
    std::cout << "[Step 1 Completed]: HostA -> DeviceA\n";

    // Sequence 2: DeviceA -> DeviceB
    // If dual GPUs exist, peer-to-peer or standard device context copying is used
    CUDA_CHECK(cudaMemcpy(deviceB, deviceA, bytes, cudaMemcpyDeviceToDevice));
    std::cout << "[Step 2 Completed]: DeviceA -> DeviceB\n";

    // Sequence 3: DeviceB -> HostB
    CUDA_CHECK(cudaMemcpy(hostB, deviceB, bytes, cudaMemcpyDeviceToHost));
    std::cout << "[Step 3 Completed]: DeviceB -> HostB\n";

    // Sequence 4: HostB -> HostC
    CUDA_CHECK(cudaMemcpy(hostC, hostB, bytes, cudaMemcpyHostToHost));
    std::cout << "[Step 4 Completed]: HostB -> HostC\n\n";

    // --- 5. Data Assertion & Integrity Inspection ---
    std::cout << "--- Final Verification ---\n";
    std::cout << "HostA Vector: ";
    for (int i = 0; i < VECTOR_SIZE; ++i) std::cout << hostA[i] << " ";
    std::cout << "\n";

    std::cout << "HostC Vector: ";
    for (int i = 0; i < VECTOR_SIZE; ++i) std::cout << hostC[i] << " ";
    std::cout << "\n\n";

    // Verification check loop
    bool integrityMatch = true;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        if (hostA[i] != hostC[i]) {
            integrityMatch = false;
            break;
        }
    }

    if (integrityMatch) {
        std::cout << "Result: SUCCESS! HostA and HostC match exactly.\n";
    } else {
        std::cout << "Result: FAILURE! Data corruption detected along the pipeline pipeline.\n";
    }

    // --- 6. Hardware/Memory Cleanup ---
    free(hostA);
    free(hostB);
    free(hostC);
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));

    return 0;
}
#include <iostream>
#include <cuda.h>

#define N 1000000

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10; i++)
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}


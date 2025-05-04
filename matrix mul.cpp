#include <iostream>
#include <cuda.h>

#define N 16

__global__ void matrixMul(int *A, int *B, int *C, int width) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    for (int i = 0; i < width; i++) {
        sum += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = sum;
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    int *d_A, *d_B, *d_C;
    int size = N * N * sizeof(int);

    // Initialize A and B
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    matrixMul<<<1, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print resulting matrix
    std::cout << "Matrix C = A x B:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << C[i][j] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}



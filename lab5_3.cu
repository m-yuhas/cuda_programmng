/* Command to compile on Windows:
nvcc .\lab5_3.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

Output should be:
A: [
[3.00, 5.00, 2.00, 0.00],
[2.00, 4.00, 5.00, 1.00],
[0.00, 3.00, 3.00, 1.00],
[3.00, 5.00, 4.00, 4.00],
[4.00, 5.00, 5.00, 3.00],
[10.00, 13.00, 21.00, 16.00],
[9.00, 11.00, 15.00, 8.00]]
b: [
[29.99],
[14.99],
[9.99],
[24.99]]
c: [
[184.90],
[194.88],
[99.93],
[304.84],
[319.83],
[1104.40],
[784.57]]
*/

#include <stdio.h>

__global__ void mat_mul(double *C, double *A, double *B) {
    extern __shared__ double tmp[];
    tmp[blockDim.y * (blockIdx.x * gridDim.y + blockIdx.y) + threadIdx.y] = A[blockIdx.x * blockDim.y + threadIdx.y] * B[threadIdx.x * gridDim.y + blockIdx.y];
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        C[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        for (int i = 0; i <= blockDim.y; i++) {
            C[blockIdx.x * gridDim.y + blockIdx.y] += tmp[blockDim.y * (blockIdx.x * gridDim.y + blockIdx.y) + i];
        }
    }
}

void mat2str(char *str, void *mat, int m, int n, int str_size) {
    double *mat_p = (double *) mat;
    int written = 0;
    written += snprintf(str + written, str_size - written, "[\n");
    for (int idx = 0; idx < m - 1; idx++) {
        written += snprintf(str + written, str_size - written, "[");
        for (int jdx = 0; jdx < n - 1; jdx++) {
            written += snprintf(str + written, str_size - written, "%.2f, ", *(mat_p + idx * n + jdx));
        }
        written += snprintf(str + written, str_size - written, "%.2f],\n", *(mat_p + idx * n + n - 1));
    }
    written += snprintf(str + written, str_size - written, "[");
    for (int jdx = 0; jdx < n - 1; jdx++) {
        written += snprintf(str + written, str_size - written, "%.2f, ", *(mat_p + (m - 1) * n + jdx));
    }
    written += snprintf(str + written, str_size - written, "%.2f]]", *(mat_p + (m - 1) * n + n - 1));
}

int main(void) {
    /* Intiialize inputs (CPU) */
    const int M = 7;
    const int N = 4;
    const int O = 1;
    double A[M][N] {
        {3, 5, 2, 0},
        {2, 4, 5, 1},
        {0, 3, 3, 1},
        {3, 5, 4, 4},
        {4, 5, 5, 3},
        {10, 13, 21, 16},
        {9, 11, 15, 8}
    };        
    double b[N][O] {
        {29.99},
        {14.99},
        {9.99},
        {24.99}
    };
    double c[M][O];
    char str_A[320];
    char str_b[320];
    mat2str(str_A, A, M, N, 320);
    mat2str(str_b, b, N, O, 320);
    printf("A: %s\n", str_A);
    printf("b: %s\n", str_b);

    /* Allocate memory for calculation on GPU */
    double *gpu_A;
    double *gpu_B;
    double *gpu_C;
    cudaMalloc((void**) &gpu_A, sizeof(double) * M * N);
    cudaMalloc((void**) &gpu_B, sizeof(double) * N * O);
    cudaMalloc((void**) &gpu_C, sizeof(double) * M * O);

    /* Copy inputs to GPU */
    cudaMemcpy(gpu_A, A, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b, sizeof(double) * N * O, cudaMemcpyHostToDevice);
 
    /* Do the thing */
    dim3 out_dim(M, O);
    dim3 block_dim(M, N);
    mat_mul<<<out_dim, block_dim, sizeof(double) * M * N * O>>>(gpu_C, gpu_A, gpu_B);
    cudaMemcpy(c, gpu_C, sizeof(double) * M * O, cudaMemcpyDeviceToHost);

    /* Remember to clean up after ourselves */
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);

    /* Print result */
    char str_c[80];
    mat2str(str_c, c, M, O, 80);
    printf("c: %s\n", str_c);
    return 0;
}

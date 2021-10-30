/* Command to compile on Windows:
nvcc .\lab5_2_2.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

Output should be:
a: [22, 13, 16, 5]
b: [5, 22, 17, 37]
c: [27, 35, 33, 42]
*/

#include <stdio.h>

__global__ void vector_add(int *c, int *a, int *b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void array2str(char *str, int *array, int n, int str_size) {
    int written = 0;
    written += snprintf(str + written, str_size - written, "[");
    for (int idx = 0; idx < n - 1; idx++) {
        written += snprintf(str + written, str_size - written, "%i, ", *(array + idx));
    }
    written += snprintf(str + written, str_size - written, "%i]", *(array + n - 1));
    return;
}

int main(void) {
    /* Intiialize inputs (CPU) */
    const int N = 4;
    int a[N] = {22, 13, 16, 5};
    int b[N] = {5, 22, 17, 37};
    int c[N];
    char str_a[80];
    char str_b[80];
    array2str(str_a, a, N, 80);
    array2str(str_b, b, N, 80);
    printf("a: %s\n", str_a);
    printf("b: %s\n", str_b);

    /* Allocate memory for calculation on GPU */
    int *gpu_a;
    int *gpu_b;
    int *gpu_c;
    cudaMalloc((void**) &gpu_a, sizeof(int) * N);
    cudaMalloc((void**) &gpu_b, sizeof(int) * N);
    cudaMalloc((void**) &gpu_c, sizeof(int) * N);

    /* Copy inputs to GPU */
    cudaMemcpy(gpu_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
 
    /* Do the thing */
    vector_add<<<1, N>>>(gpu_c, gpu_a, gpu_b);
    cudaMemcpy(c, gpu_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    /* Remember to clean up after ourselves */
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    /* Print result */
    char str_c[80];
    array2str(str_c, c, N, 80);
    printf("c: %s\n", str_c);

    return 0;
}
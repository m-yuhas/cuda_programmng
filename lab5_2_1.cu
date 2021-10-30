/* Command to compile on Windows:
nvcc .\lab5_2_1.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
*/

#include <stdio.h>

__global__ void hello_GPU(void) {
    if (blockIdx.x == 0 && threadIdx.x > 3)  {
        return;
    }
    printf("Hello from GPU%i[%i]!\n", blockIdx.x + 1, threadIdx.x);
}

int main(void) {
    printf("Hello from CPU!\n");
    hello_GPU<<<2, 6>>>();
    cudaDeviceSynchronize();
    return 0;
}
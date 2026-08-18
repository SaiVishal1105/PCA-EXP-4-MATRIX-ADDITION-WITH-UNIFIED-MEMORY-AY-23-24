# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>NAME: SAI VISHAL D</h3>
<h3>REGISTER NO: 212223230180</h3>
<h3>EX. NO: 4</h3>
<h3>DATE: 18-08-2026</h3>
<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
```
%%writefile unifmem1.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif // _COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, const int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    

// Write Your Matrix Addition Code here
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }


}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes);  );
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes););


    // initialize data at host side
    double iStart = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    double iElaps = seconds() - iStart;
    printf("initialization: \t %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    iStart = seconds();




//   Type your code here to launch your kernel
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);




    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
            grid.x, grid.y, block.x, block.y);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
```

## OUTPUT:
```
==1694== NVPROF is profiling process 1694, command: ./unifmem1
./unifmem1 Starting using Device 0: Tesla T4
Matrix size: nx 4096 ny 4096
initialization: 	 0.687925 sec
sumMatrix on host:	 0.052952 sec
sumMatrix on gpu :	 0.072872 sec <<<(128,128), (32,32)>>> 
==1694== Profiling application: ./unifmem1
==1694== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  72.837ms         2  36.419ms  414.97us  72.422ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   45.65%  191.36ms         1  191.36ms  191.36ms  191.36ms  cudaSetDevice
                   26.93%  112.87ms         1  112.87ms  112.87ms  112.87ms  cudaDeviceReset
                   17.37%  72.808ms         1  72.808ms  72.808ms  72.808ms  cudaDeviceSynchronize
                    5.05%  21.179ms         4  5.2947ms  16.859us  21.084ms  cudaMallocManaged
                    3.35%  14.045ms         4  3.5112ms  2.0429ms  5.2121ms  cudaFree
                    0.71%  2.9564ms       114  25.933us      85ns  1.3677ms  cuDeviceGetAttribute
                    0.54%  2.2459ms         1  2.2459ms  2.2459ms  2.2459ms  cudaGetDeviceProperties
                    0.40%  1.6856ms         2  842.78us  60.665us  1.6249ms  cudaLaunchKernel
                    0.00%  12.424us         1  12.424us  12.424us  12.424us  cuDeviceGetName
                    0.00%  5.2010us         2  2.6000us     148ns  5.0530us  cuDeviceGet
                    0.00%  1.6590us         1  1.6590us  1.6590us  1.6590us  cuDeviceGetPCIBusId
                    0.00%  1.0480us         3     349ns      88ns     737ns  cuDeviceGetCount
                    0.00%     885ns         1     885ns     885ns     885ns  cuDeviceTotalMem
                    0.00%     640ns         1     640ns     640ns     640ns  cudaGetLastError
                    0.00%     439ns         1     439ns     439ns     439ns  cuDeviceGetUuid
                    0.00%     325ns         1     325ns     325ns     325ns  cuModuleGetLoadingMode

==1694== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    5030  39.087KB  4.0000KB  940.00KB  192.0000MB  28.77071ms  Host To Device
     384  170.67KB  4.0000KB  0.9961MB  64.00000MB  5.816995ms  Device To Host
     297         -         -         -           -  71.99665ms  Gpu page fault groups
Total CPU Page faults: 960
```
```
==2247== NVPROF is profiling process 2247, command: ./unifmem2
./unifmem2 Starting using Device 0: Tesla T4
Matrix size: nx 4096 ny 4096
initialization: 	 0.699795 sec
sumMatrix on host:	 0.088747 sec
sumMatrix on gpu :	 0.061306 sec <<<(128,128), (32,32)>>> 
==2247== Profiling application: ./unifmem2
==2247== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  61.283ms         2  30.642ms  310.17us  60.973ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   46.27%  178.30ms         1  178.30ms  178.30ms  178.30ms  cudaSetDevice
                   27.83%  107.24ms         1  107.24ms  107.24ms  107.24ms  cudaDeviceReset
                   15.91%  61.297ms         1  61.297ms  61.297ms  61.297ms  cudaDeviceSynchronize
                    5.32%  20.493ms         4  5.1232ms  15.551us  20.400ms  cudaMallocManaged
                    3.27%  12.611ms         4  3.1527ms  1.7720ms  4.5178ms  cudaFree
                    0.73%  2.8210ms       114  24.745us      92ns  1.6332ms  cuDeviceGetAttribute
                    0.61%  2.3495ms         1  2.3495ms  2.3495ms  2.3495ms  cudaGetDeviceProperties
                    0.06%  223.94us         2  111.97us  6.6040us  217.34us  cudaLaunchKernel
                    0.00%  13.812us         1  13.812us  13.812us  13.812us  cuDeviceGetName
                    0.00%  5.2610us         2  2.6300us     465ns  4.7960us  cuDeviceGet
                    0.00%  1.9660us         1  1.9660us  1.9660us  1.9660us  cuDeviceGetPCIBusId
                    0.00%  1.2930us         3     431ns     107ns     809ns  cuDeviceGetCount
                    0.00%  1.2160us         1  1.2160us  1.2160us  1.2160us  cudaGetLastError
                    0.00%     778ns         1     778ns     778ns     778ns  cuDeviceTotalMem
                    0.00%     409ns         1     409ns     409ns     409ns  cuDeviceGetUuid
                    0.00%     325ns         1     325ns     325ns     325ns  cuModuleGetLoadingMode

==2247== Unified Memory profiling result:
Device "Tesla T4 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    3338  39.267KB  4.0000KB  876.00KB  128.0000MB  18.98123ms  Host To Device
     384  170.67KB  4.0000KB  0.9961MB  64.00000MB  5.783571ms  Device To Host
     297         -         -         -           -  60.43052ms  Gpu page fault groups
Total CPU Page faults: 768
```
## RESULT:
unifmem2 performs better than unifmem1: GPU time drops from 72.84 ms → 61.28 ms (~15.9% faster).
Host-to-device transfers reduce from 192 MB → 128 MB, and GPU page faults decrease from 960 → 768.
Overall, unifmem2 has better Unified Memory efficiency and lower memory-transfer/page-fault overhead.

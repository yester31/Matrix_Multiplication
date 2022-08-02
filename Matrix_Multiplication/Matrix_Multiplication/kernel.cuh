#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cublas_v2.h>

#define ITERS 100

// ERROR CHECK
#if defined(NDEBUG)     //release mode
#define CUDA_CHECK(x) (x)   
#else                   // debug mode
//error check 
#define CUDA_CHECK(x)   do{\
    (x); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("cuda failure %s at %s:%d \n", \
        cudaGetErrorString(e), \
            __FILE__, __LINE__); \
        exit(0); \
    } \
}while(0)
#endif

void matmult_cu(int M, int N, int K, float* mat_a, float* mat_b, float* mat_c);
void matmult_cu_shared(int M, int N, int K, float* mat_a, float* mat_b, float* mat_c);
void matmult_cublas(int M, int N, int K, float* mat_a, float* mat_b, float* mat_c);

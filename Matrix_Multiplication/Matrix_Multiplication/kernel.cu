#include "kernel.cuh"

__global__ void matMul_kernel_f32(
    float* output, 
    const float* input_a, 
    const float* input_b,
    int M, int K, int N, 
    const int tcount)
{
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos >= tcount) return;

    int w_idx = pos % N;
    int h_idx = pos / N;

    output[h_idx * N + w_idx] = 0.f;

    for (int k = 0; k < K; ++k) {
        output[h_idx * N + w_idx] += input_a[h_idx * K + k] * input_b[k * N + w_idx];
    }
}


void matmult_cu0(int M, int N, int K, float* mat_a, float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] CUDA Matrix Multiplication" << std::endl;

    //device-side data
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_o = 0;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&dev_a, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dev_o, M * N * sizeof(int)));

    uint64_t dur_time = 0;
    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    for (int i = 0; i < ITERS; i++) {

        //copy from host to device 
        CUDA_CHECK(cudaMemcpy(dev_a, mat_a, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b, mat_b, K * N * sizeof(float), cudaMemcpyHostToDevice));

        //launch a kernel on the GPU with one thread for each element.
        int thread_cnt = M * N;
        int block = 512;
        int grid = ((thread_cnt - 1) / block + 1);

        dim3 dimGrid(grid, 1, 1);
        dim3 dimBlock(block, 1, 1);

        uint64_t start_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        matMul_kernel_f32 << <dimGrid, dimBlock >> > (dev_o, dev_a, dev_b, M, K, N, thread_cnt);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        uint64_t end_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        dur_time += (end_time2 - start_time2);

        //copy from device to host
        CUDA_CHECK(cudaMemcpy(mat_c, dev_o, M * N * sizeof(int), cudaMemcpyDeviceToHost));
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    // 결과 출력
    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds] (with data transfer time)" << std::endl;
    std::cout << "[INFO] Avg elapsed time = " << (dur_time) / (ITERS) << " [milliseconds] (without data transfer time)" << std::endl;
    std::cout << "==================================================" << std::endl;
    //free device memory
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_o));
}
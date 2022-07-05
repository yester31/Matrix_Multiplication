#include "opencv2/opencv.hpp"
#include <chrono>
#include "kernel.cuh"

// 0 - 100 사이 실수로 데이터 초기화
void init_matrix(float* ptr, unsigned int size)
{
    std::cout << "[INFO] Initialization of Matrix value" << std::endl;
    srand(time(0));
    while (size--) *ptr++ = rand() % 100;
}

// 콘솔창에 행렬값을 출력
void print_matrix(std::vector<float> &output, int M, int N)
{
    std::cout << "[INFO] Print Matrix" << std::endl;
    std::cout << std::endl; std::cout << std::endl;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            std::cout << output[m * N + n] << " ";
        }std::cout << std::endl;
    }std::cout << std::endl; std::cout << std::endl;
}

// 두 행렬값 비교
void check_match(std::vector<float> &matA, std::vector<float> &matB)
{
    std::cout << "[INFO] Check Two Matrix value match" << std::endl;
    if (matA.size() != matB.size()) {
        std::cout << "[ERROR] Both Matrix size is not same!!!" << std::endl;
        return;
    }
    bool result = true;
    for (int i = 0; i < matA.size(); i++) {
        if (matA[i] != matB[i]) {
            result = false;
            break;
        }
    }
    if (result) std::cout << "[INFO] Works well~ Both Matrix is same." << std::endl;
    else std::cout << "[WARNING] Something wrong !!! Both Matrix is not same." << std::endl;
    return;
}

void matmult_cv(int M, int N, int K, float* mat_a, float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] OpenCV Matrix Multiplication" << std::endl;

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < ITERS; i++) {
        cv::Mat A = cv::Mat(M, K, CV_32FC1, mat_a);
        cv::Mat B = cv::Mat(K, N, CV_32FC1, mat_b);
        cv::Mat O;
        cv::gemm(A, B, 1, cv::Mat(), 0, O, 0);
        memcpy(mat_c, O.data, M * N * sizeof(float));
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}

void matmult(int M, int N, int K, const float* mat_a, const float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] Naive Matrix Multiplication" << std::endl;

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < ITERS; i++) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                mat_c[m * N + n] = 0;
                for (int k = 0; k < K; ++k) {
                    mat_c[m * N + n] += mat_a[m * K + k] * mat_b[k * N + n];
                }
            }
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}

void transpose(int I, int J, const float* mati, float* mato) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            mato[j * I + i] = mati[i * J + j];
        }
    }
}

void matmult_trans(int M, int N, int K, const float* mat_a, const float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] Matrix Multiplication with Transpose" << std::endl;

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<float> mat_b2(N*K);
    for (int i = 0; i < ITERS; i++) {
        transpose(K, N, mat_b, mat_b2.data());
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                mat_c[m * N + n] = 0;
                for (int k = 0; k < K; k++) {
                    mat_c[m * N + n] += mat_a[m * K + k] * mat_b2[n * K + k];
                }
            }
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}

//gcc 사용시 컴파일 명령어에 - fopenmp 를 추가
//예) gcc - g - Wall - fopenmp - o omp_ex omp_ex.c
void matmult_omp(int M, int N, int K, const float* mat_a, const float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] Matrix Multiplication with OpenMP" << std::endl;

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < ITERS; i++) {
#pragma omp parallel
        {
#pragma omp for
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    mat_c[m * N + n] = 0;
                    for (int k = 0; k < K; ++k) {
                        mat_c[m * N + n] += mat_a[m * K + k] * mat_b[k * N + n];
                    }
                }
            }
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}

void matmult_opm_trans(int M, int N, int K, const float* mat_a, const float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] Matrix Multiplication with OpenMP & Transpose" << std::endl;

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::vector<float> mat_b2(N*K);
    for (int i = 0; i < ITERS; i++) {
        transpose(K, N, mat_b, mat_b2.data());
#pragma omp parallel
        {
#pragma omp for
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    mat_c[m * N + n] = 0;
                    for (int k = 0; k < K; ++k) {
                        mat_c[m * N + n] += mat_a[m * K + k] * mat_b2[n * K + k];
                    }
                }
            }
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}

void matmult_sse(int M, int N, int K, const float* mat_a, const float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] Matrix Multiplication with SSE" << std::endl;
    if (N % 4 != 0) {
        std::cout << "[ERROR] The value of N must be a multiple of 4" << std::endl;
        return;
    }
    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < ITERS; i++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n += 4) {
                __m128 sum = _mm_load_ps(mat_c + m * N + n);
                sum = _mm_set1_ps(0);
                for (int k = 0; k < K; k++) {
                    sum = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(mat_a[m * K + k]), _mm_load_ps(mat_b + k * N + n)), sum);
                }
                _mm_store_ps(mat_c + m * N + n, sum);
            }
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}


int main()
{
    // A[M, K] * B[K, N] = C[M, N]
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    std::cout << "A[" << M << ", " << K << "] * B[" << K << ", " << N << "] = C[" << M << ", " << N << "]" << std::endl;
    std::cout << "Iteration count : " << ITERS << std::endl;

    std::vector<float> mat_a(M * K);
    std::vector<float> mat_b(K * N);
    std::vector<float> mat_c_cv(M * N);
    std::vector<float> mat_c_naive(M * N);
    std::vector<float> mat_c_trans(M * N);
    std::vector<float> mat_c_omp(M * N);
    std::vector<float> mat_c_omp_trans(M * N);
    std::vector<float> mat_c_sse(M * N);
    std::vector<float> mat_c_cu(M * N);
    std::vector<float> mat_c_shared(M * N);
    std::vector<float> mat_c_cublas(M * N);

    // initialization matrix value 
    init_matrix(mat_a.data(), mat_a.size());
    init_matrix(mat_b.data(), mat_b.size());
    //print_matrix(mat_a, M, K);
    //print_matrix(mat_b, K, N);

    // opencv matrix multiplication
    matmult_cv(M, N, K, mat_a.data(), mat_b.data(), mat_c_cv.data());

    // naive matrix multiplication 
    matmult(M, N, K, mat_a.data(), mat_b.data(), mat_c_naive.data());

    // matrix multiplication with transpose
    matmult_trans(M, N, K, mat_a.data(), mat_b.data(), mat_c_trans.data());

    // matrix multiplication with OpenMP
    matmult_omp(M, N, K, mat_a.data(), mat_b.data(), mat_c_omp.data());

    // matrix multiplication with OpenMP & transpose
    matmult_opm_trans(M, N, K, mat_a.data(), mat_b.data(), mat_c_omp_trans.data());

    // matrix multiplication with SSE
    matmult_sse(M, N, K, mat_a.data(), mat_b.data(), mat_c_sse.data());

    // matrix multiplication on GPU
    matmult_cu(M, N, K, mat_a.data(), mat_b.data(), mat_c_cu.data());

    // matrix multiplication on GPU with shared memory
    matmult_cu_shared(M, N, K, mat_a.data(), mat_b.data(), mat_c_shared.data());

    // cublas matrix multiplication on GPU
    matmult_cublas(M, N, K, mat_a.data(), mat_b.data(), mat_c_cublas.data());

    // match check
    check_match(mat_c_cv, mat_c_naive);
    check_match(mat_c_cv, mat_c_trans);
    check_match(mat_c_cv, mat_c_omp);
    check_match(mat_c_cv, mat_c_omp_trans);
    check_match(mat_c_cv, mat_c_sse);
    check_match(mat_c_cv, mat_c_cu);
    check_match(mat_c_cv, mat_c_shared);
    check_match(mat_c_cv, mat_c_cublas);

    return 0;
}
#include "opencv2/opencv.hpp"
#include <chrono>
#include "kernel.cuh"

// 0 - 100 사이 실수로 데이터 초기화
void init_matrix(float* ptr, unsigned int size)
{
    std::cout << "[INFO] Initialization of Matrix value" << std::endl;
    srand(time(0));
    while (size--) *ptr++ = rand() % 100;
    //int tt = 0;
    //while (size--) *ptr++ = ++tt;
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


// 행렬 A와 B를 더하여 C에 저장시키는 함수(A행렬 B행렬 C행렬은 같은 크기)
void MatrixSum(std::vector<float>& matrixA, std::vector<float>& matrixB, std::vector<float>& matrixC, int Rows, int Cols)
{
    for (int i = 0; i < Rows; i++)
    {
        int temp = i * Cols;
        for (int j = 0; j < Cols; j++)
        {
            int gidx = temp + j;
            matrixC[gidx] = matrixA[gidx] + matrixB[gidx];
        }
    }
}

// 행렬 A와 B를 빼서 C에 저장시키는 함수(A행렬 B행렬 C행렬은 같은 크기)
void MatrixSub(std::vector<float>& matrixA, std::vector<float>& matrixB, std::vector<float>& matrixC, int Rows, int Cols)
{
    for (int i = 0; i < Rows; i++)
    {
        int temp = i * Cols;
        for (int j = 0; j < Cols; j++)
        {
            int gidx = temp + j;
            matrixC[gidx] = matrixA[gidx] - matrixB[gidx];
        }
    }
}

// 행렬 A와 B를 곱하여 C에 저장시키는 함수(A행렬 B행렬 C행렬은 같은 크기)
// parameter : 행렬 A, B, C 
void MatrixMul(std::vector<float>& Prev_matrix, std::vector<float>& Post_matrix, std::vector<float>& Y_output, int prev_rows, int prev_colsAndPost_rows, int post_cols)
{
    for (int m = 0; m < prev_rows; ++m) {
        for (int n = 0; n < post_cols; ++n) {
            Y_output[m * post_cols + n] = 0;
            for (int k = 0; k < prev_colsAndPost_rows; ++k) {
                Y_output[m * post_cols + n] += Prev_matrix[m * prev_colsAndPost_rows + k] * Post_matrix[k * post_cols + n];
            }
        }
    }
}

void matmult2X2(int M, int N, int K, const float* mat_a, const float* mat_b, float* mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] 2X2 unlooping Matrix Multiplication" << std::endl;

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < ITERS; i++) {
        for (int m = 0; m < M/2; ++m) {
            for (int n = 0; n < N/2; ++n) {
                mat_c[m * N * 2 + n * 2] = 0;
                mat_c[m * N * 2 + n * 2 + 1] = 0;
                mat_c[m * N * 2 + n * 2 + N] = 0;
                mat_c[m * N * 2 + n * 2 + N + 1] = 0;
                for (int k = 0; k < K; ++k) {
                    mat_c[m * N * 2 + n * 2]         += mat_a[m * K * 2 + k]     * mat_b[k * N + n * 2];
                    mat_c[m * N * 2 + n * 2 + 1]     += mat_a[m * K * 2 + k]     * mat_b[k * N + n * 2 + 1];
                    mat_c[m * N * 2 + n * 2 + N]     += mat_a[m * K * 2 + k + K] * mat_b[k * N + n * 2];
                    mat_c[m * N * 2 + n * 2 + N + 1] += mat_a[m * K * 2 + k + K] * mat_b[k * N + n * 2 + 1];
                }
            }
        }
    }
    uint64_t end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::cout << "[INFO] Avg elapsed time = " << (end_time - start_time) / (ITERS) << " [milliseconds]" << std::endl;
    std::cout << "==================================================" << std::endl;
}

void MatrixMulElement4X4(std::vector<float>& matrix_U, std::vector<float>& matrix_V, std::vector<float>& matrix_M, int U_rows, int UcolsVrows, int V_cols)
{
    for (int i = 0; i < U_rows / 4; ++i) { // 행
        for (int j = 0; j < V_cols / 4; ++j) // 열 
        {
            int m1 = 0; int m2 = 0; int m3 = 0; int m4 = 0; int m5 = 0; int m6 = 0; int m7 = 0; int m8 = 0;
            int m9 = 0; int m10 = 0; int m11 = 0; int m12 = 0; int m13 = 0; int	m14 = 0; int m15 = 0; int m16 = 0;
            for (int k = 0; k < UcolsVrows / 4; ++k)
            {
                int midx1 = i * UcolsVrows * 4 + k * 4;
                int midx2 = k * V_cols * 4 + j * 4;
                m1 += matrix_U[midx1] * matrix_V[midx2];
                m2 += matrix_U[midx1 + 1] * matrix_V[midx2 + 1];
                m3 += matrix_U[midx1 + 2] * matrix_V[midx2 + 2];
                m4 += matrix_U[midx1 + 3] * matrix_V[midx2 + 3];

                int midx3 = midx1 + UcolsVrows;
                int midx4 = midx2 + V_cols;
                m5 += matrix_U[midx3] * matrix_V[midx4];
                m6 += matrix_U[midx3 + 1] * matrix_V[midx4 + 1];
                m7 += matrix_U[midx3 + 2] * matrix_V[midx4 + 2];
                m8 += matrix_U[midx3 + 3] * matrix_V[midx4 + 3];

                int midx5 = midx3 + UcolsVrows;
                int midx6 = midx4 + V_cols;
                m9 += matrix_U[midx5] * matrix_V[midx6];
                m10 += matrix_U[midx5 + 1] * matrix_V[midx6 + 1];
                m11 += matrix_U[midx5 + 2] * matrix_V[midx6 + 2];
                m12 += matrix_U[midx5 + 3] * matrix_V[midx6 + 3];

                int midx7 = midx5 + UcolsVrows;
                int midx8 = midx6 + V_cols;
                m13 += matrix_U[midx7] * matrix_V[midx8];
                m14 += matrix_U[midx7 + 1] * matrix_V[midx8 + 1];
                m15 += matrix_U[midx7 + 2] * matrix_V[midx8 + 2];
                m16 += matrix_U[midx7 + 3] * matrix_V[midx8 + 3];
            }
            int idx1 = i * V_cols * 4 + j * 4;
            matrix_M[idx1] = m1;
            matrix_M[idx1 + 1] = m2;
            matrix_M[idx1 + 2] = m3;
            matrix_M[idx1 + 3] = m4;

            int idx2 = idx1 + V_cols;
            matrix_M[idx2] = m5;
            matrix_M[idx2 + 1] = m6;
            matrix_M[idx2 + 2] = m7;
            matrix_M[idx2 + 3] = m8;

            int idx3 = idx2 + V_cols;
            matrix_M[idx3] = m9;
            matrix_M[idx3 + 1] = m10;
            matrix_M[idx3 + 2] = m11;
            matrix_M[idx3 + 3] = m12;

            int idx4 = idx3 + V_cols;
            matrix_M[idx4] = m13;
            matrix_M[idx4 + 1] = m14;
            matrix_M[idx4 + 2] = m15;
            matrix_M[idx4 + 3] = m16;
        }
    }
}

// 임계값 구하는 함수
int getThreshold(int n)
{
    int th;
    double k = floor(log(n) / log(2) - 6);
    th = (int)floor(n / pow(2.0, k)) + 1;
    return th;
}

// 4개의 부분행렬로 나누는 함수
// parameter : 나눌 행렬, 저장할 행렬 공간 4개
void Submatrix(std::vector<float>& matrixOrigin, std::vector<float>& matrix11, std::vector<float>& matrix12, std::vector<float>& matrix21, std::vector<float>& matrix22, int Rows, int Cols)
{
    // int Rows, int Cols 부분행렬의 사이즈 
    for (int i = 0; i < Rows; i++)
    {
        int temp = i * (Cols * 2);
        int temp2 = i * (Cols);
        for (int j = 0; j < Cols; j++)
        {
            int gidx = temp + j;
            int gidx2 = temp2 + j;
            matrix11[gidx2] = matrixOrigin[gidx];									//좌 상단행렬
            matrix12[gidx2] = matrixOrigin[gidx + Cols];							//우 상단행렬
            matrix21[gidx2] = matrixOrigin[Rows * Cols * 2 + gidx];					//좌 하단행렬
            matrix22[gidx2] = matrixOrigin[Rows * Cols * 2 + gidx + Cols];			//우 하단행렬
        }
    }
}

// 4개의 부분행렬들을 재결합 해주는 함수
// parameter : 합친 결과를 저장할 행렬 , 부분행렬 11, 12, 21, 22
void Mergematrix(std::vector<float>& matrixOrigin, std::vector<float>& matrix11, std::vector<float>& matrix12, std::vector<float>& matrix21, std::vector<float>& matrix22, int Rows, int Cols)
{
    for (int i = 0; i < Rows; i++)
    {
        int temp = i * (Cols * 2);
        int temp2 = i * Cols;
        for (int j = 0; j < Cols; j++)
        {
            int gidx = temp + j;
            int gidx2 = temp2 + j;
            matrixOrigin[gidx] = matrix11[gidx2];										//좌 상단행렬
            matrixOrigin[gidx + Cols] = matrix12[gidx2];								//우 상단행렬
            matrixOrigin[Rows * Cols * 2 + gidx] = matrix21[gidx2];         			//좌 하단행렬
            matrixOrigin[Rows * Cols * 2 + gidx + Cols] = matrix22[gidx2];				//우 하단행렬
        }
    }
}


// 쉬트라센 알고리즘 함수
void Strassen(std::vector<float>& matrixU, std::vector<float>& matrixV, std::vector<float>& matrixM, int U_Row, int U_Col, int V_Row, int V_Col)
{
    if (V_Col <= getThreshold(V_Col)) {
        MatrixMul(matrixU, matrixV, matrixM, U_Row, U_Col, V_Col);
        return;
    }
    else {
        int newU_Row = U_Row / 2; //4등분을 하기 위해
        int newU_Col = U_Col / 2;
        int newV_Row = V_Row / 2;
        int newV_Col = V_Col / 2;

        //a11~a22 부분행렬, b11~b22 부분행렬 
        std::vector<float> a11(newU_Row * newU_Col), a12(newU_Row * newU_Col), a21(newU_Row * newU_Col), a22(newU_Row * newU_Col);
        std::vector<float> b11(newV_Row * newV_Col), b12(newV_Row * newV_Col), b21(newV_Row * newV_Col), b22(newV_Row * newV_Col);

        //부분행렬들의 연산결과를 m1~m7 에 저장
        std::vector<float>  m1(newU_Row * newV_Col), m2(newU_Row * newV_Col), m3(newU_Row * newV_Col), m4(newU_Row * newV_Col), m5(newU_Row * newV_Col), m6(newU_Row * newV_Col), m7(newU_Row * newV_Col);

        //a11~b22 의 연산결과들을 임시로 저장할 그릇
        std::vector<float>  tempA(newU_Row * newU_Col), tempB(newV_Row * newV_Col);
        std::vector<float>  tempAc(newU_Row * newV_Col), tempBc(newU_Row * newV_Col);

        // m1~m7 연산 결과로 C를 구하기 위해 저장 할 행렬
        std::vector<float>  c11(newU_Row * newV_Col), c12(newU_Row * newV_Col), c21(newU_Row * newV_Col), c22(newU_Row * newV_Col);

        //A의 부분행렬 4개, B의 부분행렬 4개 생성
        Submatrix(matrixU, a11, a12, a21, a22, newU_Row, newU_Col);
        Submatrix(matrixV, b11, b12, b21, b22, newV_Row, newV_Col);

        MatrixSum(a11, a22, tempA, newU_Row, newU_Col);                     // a11+a22
        MatrixSum(b11, b22, tempB, newV_Row, newV_Col);                     // b11+b22
        Strassen(tempA, tempB, m1, newU_Row, newU_Col, newV_Row, newV_Col); // m1=(a11+a11)(b11+b22)

        MatrixSum(a21, a22, tempA, newU_Row, newU_Col);                   // a21+a22
        Strassen(tempA, b11, m2, newU_Row, newU_Col, newV_Row, newV_Col); // m2=(a21+a22)b11

        MatrixSub(b12, b22, tempB, newV_Row, newV_Col);                   // b12-b22
        Strassen(a11, tempB, m3, newU_Row, newU_Col, newV_Row, newV_Col); // m3=a11(b12-b22)

        MatrixSub(b21, b11, tempB, newV_Row, newV_Col);                   // b21-b11
        Strassen(a22, tempB, m4, newU_Row, newU_Col, newV_Row, newV_Col); // m4=a22(b21-11)

        MatrixSum(a11, a12, tempA, newU_Row, newU_Col);                   //  a11+a12
        Strassen(tempA, b22, m5, newU_Row, newU_Col, newV_Row, newV_Col); // m5=(a11+a12)b22

        MatrixSub(a21, a11, tempA, newU_Row, newU_Col);                     // a21-a11
        MatrixSum(b11, b12, tempB, newV_Row, newV_Col);                     // b11+b12
        Strassen(tempA, tempB, m6, newU_Row, newU_Col, newV_Row, newV_Col); // m6=(a21-a11)(b11+b12)

        MatrixSub(a12, a22, tempA, newU_Row, newU_Col);                     // a12-a22
        MatrixSum(b21, b22, tempB, newV_Row, newV_Col);                     // b21+b22
        Strassen(tempA, tempB, m7, newU_Row, newU_Col, newV_Row, newV_Col); // m7 = (a12 - a22)(a12 - a22)

        // 위에서 계산된 m1~m7 결과로  c11 ~ c22 를 만든다.
        MatrixSum(m1, m4, tempAc, newU_Row, newV_Col);     //m1 + m4
        MatrixSum(tempAc, m7, tempBc, newU_Row, newV_Col); //m1 + m4 + m7
        MatrixSub(tempBc, m5, c11, newU_Row, newV_Col);    //c11 = m1 + m4 - m5 + m7

        MatrixSum(m3, m5, c12, newU_Row, newV_Col);        //c12 = m3 + m5

        MatrixSum(m2, m4, c21, newU_Row, newV_Col);        //c21 = m2 + m4

        MatrixSum(m1, m3, tempAc, newU_Row, newV_Col);     //m1 + m3
        MatrixSum(tempAc, m6, tempBc, newU_Row, newV_Col); //m1 + m3 + m6
        MatrixSub(tempBc, m2, c22, newU_Row, newV_Col);    //c22 = m1 + m3 - m2 + m6

        //재병합
        Mergematrix(matrixM, c11, c12, c21, c22, newU_Row, newV_Col);
    }
}

bool twoN(int num)
{
    return (num & (num - 1)) == 0;
}

void matmult_strassen(int M, int N, int K, std::vector<float>& mat_a, std::vector<float>& mat_b, std::vector<float>& mat_c)
{
    std::cout << "==================================================" << std::endl;
    std::cout << "[INFO] Strassen Matrix Multiplication" << std::endl;

    if (N != K || M != K) {
        std::cout << "[ERROR] Input matrixs must be square matrix." << std::endl;
        return;
    }

    if (!twoN(N)) {
        std::cout << "[ERROR] Input matrix size must be 2 to the n power." << std::endl;
    }

    uint64_t start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < ITERS; i++) {

        Strassen(mat_a, mat_b, mat_c, M, K, K, N);

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
    const int M = 512;
    const int K = 512;
    const int N = 512;

    std::cout << "A[" << M << ", " << K << "] * B[" << K << ", " << N << "] = C[" << M << ", " << N << "]" << std::endl;
    std::cout << "Iteration count : " << ITERS << std::endl;

    std::vector<float> mat_a(M * K);
    std::vector<float> mat_b(K * N);
    std::vector<float> mat_c_cv(M * N);
    std::vector<float> mat_c_naive(M * N);
    std::vector<float> mat_c_2x2(M * N);
    std::vector<float> mat_c_strassen(M * N);
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
    //print_matrix(mat_c_cv, M, N);

    // naive matrix multiplication 
    matmult(M, N, K, mat_a.data(), mat_b.data(), mat_c_naive.data());

    // 2X2 unlooping matrix multiplication 
    matmult2X2(M, N, K, mat_a.data(), mat_b.data(), mat_c_2x2.data());
    //print_matrix(mat_c_2x2, M, N);

    // strassen matrix multiplication
    matmult_strassen(M, N, K, mat_a, mat_b, mat_c_strassen);

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
    check_match(mat_c_cv, mat_c_2x2);
    check_match(mat_c_cv, mat_c_strassen);
    check_match(mat_c_cv, mat_c_trans);
    check_match(mat_c_cv, mat_c_omp);
    check_match(mat_c_cv, mat_c_omp_trans);
    check_match(mat_c_cv, mat_c_sse);
    check_match(mat_c_cv, mat_c_cu);
    check_match(mat_c_cv, mat_c_shared);
    check_match(mat_c_cv, mat_c_cublas);

    return 0;
}
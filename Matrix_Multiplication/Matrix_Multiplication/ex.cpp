////나이브 코드 예시
//
//#include <stdio.h>
//#include <time.h>
//#include <stdlib.h>
//#include <vector>
//
//class Timer {
//    struct timespec s_;
//public:
//    Timer() { tic(); }
//    void tic() {
//        clock_gettime(CLOCK_REALTIME, &s_);
//    }
//
//    double toc() {
//        struct timespec e;
//        clock_gettime(CLOCK_REALTIME, &e);
//        return (double)(e.tv_sec - s_.tv_sec) + 1e-9 * (double)(e.tv_nsec - s_.tv_nsec);
//    }
//};
//
//// Please optimize this function
//void matmult(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c)
//{
//    /*
//        == input ==
//        mat_a: m x k matrix
//        mat_b: k x n matrix
//
//        == output ==
//        mat_c: m x n matrix (output)
//    */
//
//    for (int i1 = 0; i1 < m; i1++) {
//        for (int i2 = 0; i2 < n; i2++) {
//            mat_c[n*i1 + i2] = 0;
//            for (int i3 = 0; i3 < k; i3++) {
//                mat_c[n*i1 + i2] += mat_a[i1 * k + i3] * mat_b[i3 * n + i2];
//            }
//        }
//    }
//}
//
//void genmat(int n, int m, std::vector<float>& mat)
//{
//    srand(time(0));
//    mat.resize(n * m);
//    for (int i = 0; i < mat.size(); i++) mat[i] = rand() % 100;
//}
//
//void dumpmat(int n, int m, std::vector<float>& mat)
//{
//    for (int i = 0; i < n; i++)
//    {
//        for (int j = 0; j < m; j++)
//            printf("%f ", mat[i * m + j]);
//        printf("\n");
//    }
//}
//
//int ex_main(int argc, char** argv)
//{
//    std::vector<float> mat_a;
//    std::vector<float> mat_b;
//    std::vector<float> mat_c;
//
//    genmat(10, 10, mat_a);
//    genmat(10, 10, mat_b);
//    genmat(10, 10, mat_c);
//
//    Timer t;
//    double elapsed = 0;
//    const int iteration = 10000;
//    for (int i = 0; i < iteration; i++)
//    {
//        t.tic();
//        matmult(10, 10, 10, &mat_a[0], &mat_b[0], &mat_c[0]);
//        elapsed += t.toc();
//    }
//
//    dumpmat(10, 10, mat_a);
//    dumpmat(10, 10, mat_c);
//    printf("%lf ms\n", 1000.0 * elapsed / iteration);
//    return 0;
//}

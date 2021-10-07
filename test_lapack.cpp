#include <Eigen/Eigen>
#include <iostream>
#include <egl/readDMAT.h>

extern "C" int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

using namespace std;
using namespace Eigen;

void test1()
{
    double A[9] = {4, -2, 2, -2, 2, -4, 2, -4, 11};
    char uplo = 'U';
    int n = 3;
    int info;

    dpotrf_(&uplo, &n, A, &n, &info);
    for (int i = 0; i < 9; ++i)
    {
        printf("%g ", A[i]);
        if (i % 3 == 2)
            printf("\n");
    }
}

void test2()
{
    int n = 64;
    Eigen::MatrixXd A = MatrixXd::Random(n, n);
    // cout << "Original:\n"
    //      << A << endl;
    char uplo = 'U';
    int info;

    dpotrf_(&uplo, &n, &A(0, 0), &n, &info);
    cout << "U" << endl;
    cout << A << endl;
}

void test3()
{
    MatrixXd U, A;
    egl::readDMAT("Ujj.mat", A);
#pragma omp parallel for
    for (int i = 0; i < 8; ++i)
    {
        U = A;
        // U.resize(3, 3);
        // U.setZero();
        // U(0, 0) = 2;
        // U(0, 1) = -1;
        // U(0, 2) = 1;
        // U(1, 1) = 1;
        // U(1, 2) = -3;
        // U(2, 2) = 1;
        // A = U.transpose() * U;
        // cout << A << endl;
        int numJ = U.rows();
        // for (int j = 0; j < numJ; ++j)
        // {
        //     cout << "col" << j << ": " << U.col(j).transpose() << "\n\n";
        // }
        char uplo = 'U';
        int info;
        U = A;
        dpotrf_(&uplo, &numJ, U.data(), &numJ, &info);
        // cout << "dense cholesky info=" << info << endl;
        // for (int j = 0; j < numJ; ++j)
        // {
        //     cout << "col" << j << ": " << U.col(j).transpose() << "\n\n";
        // }
        MatrixXd R = U.triangularView<Upper>();
        // cout << R << endl;
        cout << i << ": " << (A - R.transpose() * R).squaredNorm() << endl;
    }
}

int main()
{
    // test1();
    // test2();
    test3();
    return 0;
}
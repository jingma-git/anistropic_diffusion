#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
using namespace std;
// Eigen Sparse Matrix tutorial

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void insertCoefficient(int id, int i, int j, double w, std::vector<T> &coeffs,
                       Eigen::VectorXd &b, const Eigen::VectorXd &boundary)
{
    int n = int(boundary.size());
    int id1 = i + j * n;

    if (i == -1 || i == n)
        b(id) -= w * boundary(j); // constrained coefficient
    else if (j == -1 || j == n)
        b(id) -= w * boundary(i); // constrained coefficient
    else
        coeffs.push_back(T(id, id1, w)); // unknown coefficient
}

void buildProblem(std::vector<T> &coefficients, Eigen::VectorXd &b, int n)
{
    b.setZero();
    Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, M_PI).sin().pow(2);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            int id = i + j * n;
            insertCoefficient(id, i - 1, j, -1, coefficients, b, boundary);
            insertCoefficient(id, i + 1, j, -1, coefficients, b, boundary);
            insertCoefficient(id, i, j - 1, -1, coefficients, b, boundary);
            insertCoefficient(id, i, j + 1, -1, coefficients, b, boundary);
            insertCoefficient(id, i, j, 4, coefficients, b, boundary);
        }
    }
}

void saveImg(const Eigen::VectorXd &x, int n, const char *fname)
{
    Eigen::Array<unsigned char, -1, -1> bits = (x * 255).cast<unsigned char>();
    cv::Mat img(n, n, CV_8UC1, bits.data());
    cv::imwrite(fname, img);
}

int main()
{
    int n = 300;
    int m = n * n;
    std::vector<T> coefficients;
    Eigen::VectorXd b(m);
    buildProblem(coefficients, b, n);

    SpMat A(m, m);
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    Eigen::SimplicialCholesky<SpMat> chol(A);
    Eigen::VectorXd x = chol.solve(b);
    saveImg(x, n, "output/laplace.png");
    return 0;
}
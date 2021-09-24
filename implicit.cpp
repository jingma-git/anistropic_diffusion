#include <vector>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include "util.h"
using namespace std;

// approximate the derivative using the new mesh instead of old mesh,
// get to the stability state faster
// (I-lambda*dt*L) * Xt+1 = Xt eq9 in "implicit fairing"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

double lambda = 0.1;
bool isImplicit = true;
const double K = 10;
const double K2 = 1 / K / K;

void insertCoefficient(int id, int i, int j, double w, std::vector<T> &coeffs,
                       int rows, int cols)
{
    int id1 = i * cols + j;
    if (i >= 0 && i < rows && j >= 0 && j < cols)
    {
        // cout << id << "," << id1 << ": " << w << endl;
        coeffs.push_back(T(id, id1, w)); // unknown coefficient
        coeffs.push_back(T(id, id, -w));
    }
}

// C: diffusion coeff
void buildProblem(const Eigen::MatrixXd &C, std::vector<T> &coefficients, int rows, int cols)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            int id = i * cols + j;

            double Cp = C(i, j);
            double Cn = (i == 0) ? 0 : C(i - 1, j);
            double Cs = (i == rows - 1) ? 0 : C(i + 1, j);
            double Ce = (j == cols - 1) ? 0 : C(i, j + 1);
            double Cw = (j == 0) ? 0 : C(i, j - 1);

            insertCoefficient(id, i - 1, j, Cn + Cp, coefficients, rows, cols);
            insertCoefficient(id, i + 1, j, Cs + Cp, coefficients, rows, cols);
            insertCoefficient(id, i, j - 1, Cw + Cp, coefficients, rows, cols);
            insertCoefficient(id, i, j + 1, Ce + Cp, coefficients, rows, cols);
        }
    }
}

void saveImg(const Eigen::VectorXd &x, int rows, int cols, const std::string fname)
{
    Eigen::Array<unsigned char, -1, -1> bits = (x).cast<unsigned char>();
    cv::Mat img(rows, cols, CV_8UC1, bits.data());
    cv::imwrite(fname, img);
}

void spIdentity(int n, Eigen::SparseMatrix<double> &I)
{
    std::vector<T> data(n);
    for (int i = 0; i < n; ++i)
    {
        data.emplace_back(i, i, 1.0);
    }
    I.resize(n, n);
    I.setFromTriplets(data.begin(), data.end());
}

void grad_debug(double *ptr, int rows, int cols, Eigen::MatrixXd &C) // d: diffusion coeff
{
    Eigen::Map<Eigen::MatrixXd> It(ptr, rows, cols);
    Eigen::MatrixXd Gx, Gy, G;
    Gx = Gy = G = C = Eigen::MatrixXd::Zero(rows, cols);
    for (int i = 1; i < rows - 1; ++i)
    {
        for (int j = 1; j < cols - 1; ++j)
        {
            // gradientX, sobel
            Gx(i, j) = fabs(It(i - 1, j + 1) - It(i - 1, j - 1) +
                            2 * It(i, j + 1) - 2 * It(i, j - 1) +
                            It(i + 1, j + 1) - It(i + 1, j - 1));

            Gy(i, j) = fabs(It(i + 1, j - 1) - It(i - 1, j - 1) +
                            2 * It(i + 1, j) - 2 * It(i - 1, j) +
                            It(i + 1, j + 1) - It(i - 1, j + 1));

            G(i, j) = Gx(i, j) * Gx(i, j) + Gy(i, j) * Gy(i, j);
            C(i, j) = 1.0 / (1 + G(i, j) * 0.01);
        }
    }
    // cout << "dx\n"
    //      << Gx << endl;
    // cout << "dy\n"
    //      << Gy << endl;
    // Eigen::Array<unsigned char, -1, -1> bitsX = Gx.cast<unsigned char>();
    // Eigen::Array<unsigned char, -1, -1> bitsY = Gy.cast<unsigned char>();
    // cv::Mat sobelX(rows, cols, CV_8UC1, bitsX.data());
    // cv::Mat sobelY(rows, cols, CV_8UC1, bitsY.data());
    // cv::imshow("sobelX", sobelX);
    // cv::imshow("sobelY", sobelY);
    // cv::waitKey(-1);
}

void grad(double *ptr, int rows, int cols, Eigen::MatrixXd &C) // d: diffusion coeff
{
    cv::Mat It(rows, cols, CV_64F, ptr);
    C = Eigen::MatrixXd::Zero(rows, cols);
    cv::Mat dx, dy;
    cv::Sobel(It, dx, CV_64F, 1, 0, 3); // gradient along x
    cv::Sobel(It, dy, CV_64F, 0, 1, 3); // gradient along y
    for (int i = 0; i < It.rows; ++i)
        for (int j = 0; j < It.cols; ++j)
        {
            double gx = dx.at<double>(i, j), gy = dy.at<double>(i, j);
            double c;
            if (i == 0 || i == It.rows - 1 || j == 0 || j == It.cols - 1)
                c = 0; // no diffusion on boundary
            else
            {
                // c = std::exp(-(gx * gx + gy * gy) * K2);
                c = 1.0 / (1.0 + (gx * gx + gy * gy) * K2);
            }
            C(i, j) = c;
        }
}

namespace fs = boost::filesystem;
int main()
{
    cv::Mat img = cv::imread("data/shell.png", cv::IMREAD_GRAYSCALE);
    string outdir = "output/";
    outdir += (isImplicit ? "implicit/lam" : "explicit/lam") + to_string(lambda) + "/";
    if (!fs::exists(outdir))
        fs::create_directories(outdir);

    cv::imwrite(outdir + "img.png", img);
    // cv::Mat img = cv::Mat(3, 3, CV_8UC1);
    // std::iota(img.data, img.data + 9, 0);
    // cout << img << endl;
    cv::Mat I0;
    img.convertTo(I0, CV_64F);

    int dof = img.rows * img.cols;
    Eigen::Map<Eigen::VectorXd> x0((double *)I0.data, dof, 1);

    SpMat I;
    spIdentity(dof, I);
    Eigen::VectorXd x = x0;
    for (int iter = 0; iter <= 1000; ++iter) // 0.053198s for explicit, 3.259856s for implicit
    {
        auto start = get_time();
        Eigen::MatrixXd C;
        grad(x.data(), img.rows, img.cols, C);
        std::vector<T> coefficients;
        buildProblem(C, coefficients, img.rows, img.cols);
        SpMat L(dof, dof);
        L.setFromTriplets(coefficients.begin(), coefficients.end());

        if (isImplicit)
        {
            SpMat S = I - lambda * L;
            Eigen::SimplicialCholesky<SpMat> chol(S); // in implicit schema, lambda can be set to 1.0
            x = chol.solve(x);
        }
        else
        {
            x += lambda * L * x;
        }

        printf("iter%d time=%f\n", iter, get_time() - start);
        if (iter % 100 == 0)
        {
            saveImg(x, img.rows, img.cols, outdir + to_string(iter) + ".png");
        }
    }

    return 0;
}
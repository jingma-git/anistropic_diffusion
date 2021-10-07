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

typedef float Float;
typedef Eigen::SparseMatrix<Float> SpMat; // declares a column-major sparse matrix type of Float
typedef Eigen::Triplet<Float> T;
typedef Eigen::Matrix<Float, -1, -1> mat_t;
typedef Eigen::Matrix<Float, -1, 1> vec_t;

int float_bytes = CV_32F; // 2^5 32bits
Float lambda = 1000.0;
bool isImplicit = true;
const Float K = 10;
const Float K2 = 1 / K / K;
const int mod = lambda > 10 ? 1 : 1.0 / (0.1 * lambda);
const int max_iter = lambda >= 10 ? 10 : mod * 10;

void insertCoefficient(int id, int i, int j, Float w, std::vector<T> &coeffs,
                       int rows, int cols)
{
    int id1 = i * cols + j;
    if (i >= 0 && i < rows && j >= 0 && j < cols)
    {
        // cout << id << "," << id1 << ": " << w << endl;
        coeffs.push_back(T(id, id1, w)); // expensive compared with explicit counterpart
        coeffs.push_back(T(id, id, -w));
    }
}

// C: diffusion coeff
void buildProblem(mat_t &C, std::vector<T> &coefficients, int rows, int cols)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            int id = i * cols + j;

            Float Cp = C(i, j);
            Float Cn = (i == 0) ? 0 : C(i - 1, j);
            Float Cs = (i == rows - 1) ? 0 : C(i + 1, j);
            Float Ce = (j == cols - 1) ? 0 : C(i, j + 1);
            Float Cw = (j == 0) ? 0 : C(i, j - 1);

            insertCoefficient(id, i - 1, j, Cn + Cp, coefficients, rows, cols);
            insertCoefficient(id, i + 1, j, Cs + Cp, coefficients, rows, cols);
            insertCoefficient(id, i, j - 1, Cw + Cp, coefficients, rows, cols);
            insertCoefficient(id, i, j + 1, Ce + Cp, coefficients, rows, cols);

            // Add additional two links
            Float Cne = (i == 0 || j == cols - 1) ? 0 : C(i - 1, j + 1);
            Float Csw = (i == rows - 1 || j == 0) ? 0 : C(i + 1, j - 1);
            insertCoefficient(id, i - 1, j + 1, Cne + Cp, coefficients, rows, cols);
            insertCoefficient(id, i + 1, j - 1, Csw + Cp, coefficients, rows, cols);
        }
    }
}

void saveImg(const vec_t &x, int rows, int cols, const std::string fname)
{
    Eigen::Array<unsigned char, -1, -1> bits = (x).cast<unsigned char>();
    cv::Mat img(rows, cols, CV_8UC1, bits.data());
    cv::imwrite(fname, img);
}

void spIdentity(int n, Eigen::SparseMatrix<Float> &I)
{
    std::vector<T> data(n);
    for (int i = 0; i < n; ++i)
    {
        data.emplace_back(i, i, 1.0);
    }
    I.resize(n, n);
    I.setFromTriplets(data.begin(), data.end());
}

void grad_debug(Float *ptr, int rows, int cols, mat_t &C) // d: diffusion coeff
{
    Eigen::Map<mat_t> It(ptr, rows, cols);
    mat_t Gx, Gy, G;
    Gx = Gy = G = C = mat_t::Zero(rows, cols);
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

void grad(Float *ptr, int rows, int cols, mat_t &C) // d: diffusion coeff
{
    cv::Mat It(rows, cols, float_bytes, ptr);
    C = mat_t::Zero(rows, cols);
    cv::Mat dx, dy;

    cv::Sobel(It, dx, float_bytes, 1, 0, 3); // gradient along x
    cv::Sobel(It, dy, float_bytes, 0, 1, 3); // gradient along y
    for (int i = 0; i < It.rows; ++i)
        for (int j = 0; j < It.cols; ++j)
        {
            Float gx = dx.at<Float>(i, j), gy = dy.at<Float>(i, j);
            Float c;
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
    if (sizeof(Float) == 8)
    {
        float_bytes = CV_64F;
    }
    cout << "float_bytes:" << float_bytes << " size:" << sizeof(Float) << " max_iter=" << max_iter << endl;
    cv::Mat img = cv::imread("data/shell.png", cv::IMREAD_GRAYSCALE);
    // cv::resize(img, img, cv::Size(1280, 800));
    // cv::imwrite(outdir + "img.png", img);

    string outdir = "output/" + to_string(float_bytes) + "/";
    outdir += (isImplicit ? "implicit/lam" : "explicit/lam") + to_string(lambda) + "/";
    if (!fs::exists(outdir))
        fs::create_directories(outdir);

    cv::imwrite(outdir + "img.png", img);

    cv::Mat I0;
    img.convertTo(I0, float_bytes);

    int dof = img.rows * img.cols;
    Eigen::Map<vec_t> x0((Float *)I0.data, dof, 1);

    SpMat I;
    spIdentity(dof, I);
    vec_t x = x0;
    for (int iter = 0; iter <= max_iter; ++iter) // 0.053198s for explicit, 3.259856s for implicit
    {
        auto start = get_time();
        mat_t C;
        grad(x.data(), img.rows, img.cols, C);
        std::vector<T> coefficients(img.rows * img.cols * 4);
        buildProblem(C, coefficients, img.rows, img.cols);
        SpMat L(dof, dof);
        L.setFromTriplets(coefficients.begin(), coefficients.end());
        auto build_time = get_time() - start;
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

        printf("iter%d time=%f build_time=%f\n", iter, get_time() - start, build_time);
        if (iter % mod == 0)
        {
            saveImg(x, img.rows, img.cols, outdir + to_string(iter) + ".png");
        }
    }

    return 0;
}
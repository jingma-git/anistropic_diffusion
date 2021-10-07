#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include "ichol.h"
#include "util.h"
using namespace ichol;
using namespace std;

void spIdentity(int n, Eigen::SparseMatrix<Float> &I)
{
    std::vector<Eigen::Triplet<Float>> data(n);
    for (int i = 0; i < n; ++i)
    {
        data.emplace_back(i, i, 1.0);
    }
    I.resize(n, n);
    I.setFromTriplets(data.begin(), data.end());
}

void saveImg(const vecd_t &x, int rows, int cols, const std::string fname)
{
    Eigen::Array<unsigned char, -1, -1> bits = (x).cast<unsigned char>();
    cv::Mat img(rows, cols, CV_8UC1, bits.data());
    cv::imwrite(fname, img);
}

class img_poisson
{
public:
    // algorithm parameters
    Float lambda = 100.0;
    const Float K = 10;
    const Float K2 = 1 / K / K;

public:
    img_poisson(const mati_t &cells, const matd_t &nods, int rows, int cols)
        : cels_(cells), nods_(nods), rows_(rows), cols_(cols), dim_(rows * cols)
    {
        spIdentity(dim_, I_);
    }

    size_t dim() const
    {
        return dim_;
    }

    void LHS(const Float *u, Eigen::SparseMatrix<Float> &A)
    {
        int rows = rows_;
        int cols = cols_;
        matd_t C(rows, cols);
        grad(u, rows, cols, C);
        std::vector<Eigen::Triplet<Float>> coeffs;
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

                insertCoefficient(id, i - 1, j, Cn + Cp, coeffs, rows, cols);
                insertCoefficient(id, i + 1, j, Cs + Cp, coeffs, rows, cols);
                insertCoefficient(id, i, j - 1, Cw + Cp, coeffs, rows, cols);
                insertCoefficient(id, i, j + 1, Ce + Cp, coeffs, rows, cols);
            }
        }
        Eigen::SparseMatrix<Float> L(dim_, dim_);
        L.setFromTriplets(coeffs.begin(), coeffs.end());
        A = I_ - lambda * L;
    }

    void RHS(const Float *u, vecd_t &b) const
    {
        b = Eigen::Map<const vecd_t>(u, dim_, 1);
    }

private:
    void grad(const Float *ptr, int rows, int cols, matd_t &C) // d: diffusion coeff
    {
        cv::Mat It(rows, cols, sizeof(Float) == 4 ? CV_32F : CV_64F, const_cast<Float *>(ptr)); // image at time step t
        C = matd_t::Zero(rows, cols);
        cv::Mat dx, dy;

        cv::Sobel(It, dx, sizeof(Float) == 4 ? CV_32F : CV_64F, 1, 0, 3); // gradient along x
        cv::Sobel(It, dy, sizeof(Float) == 4 ? CV_32F : CV_64F, 0, 1, 3); // gradient along y
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

    void insertCoefficient(int id, int i, int j, Float w, std::vector<Eigen::Triplet<Float>> &coeffs,
                           int rows, int cols)
    {
        int id1 = i * cols + j;
        if (i >= 0 && i < rows && j >= 0 && j < cols)
        {
            // cout << id << "," << id1 << ": " << w << endl;
            coeffs.emplace_back(id, id1, w); // expensive compared with explicit counterpart
            coeffs.emplace_back(id, id, -w);
        }
    }

private:
    const mati_t &cels_;
    const matd_t &nods_;
    int rows_, cols_, dim_;
    Eigen::SparseMatrix<Float> I_;
};

int main()
{
    string outdir = "output/smooth2d/";

    cv::Mat img = cv::imread("data/shell.png", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(1280, 800));
    cv::Mat I0(img.rows, img.cols, sizeof(Float) == 4 ? CV_32F : CV_64F);
    img.convertTo(I0, sizeof(Float) == 4 ? CV_32F : CV_64F); // 4 bytes == 32 bits
    // cv::Mat I0(3, 3, CV_32F);
    // std::fill(I0.data, I0.data + 9, 0);
    // cout << I0 << endl;
    int num_nods = I0.rows * I0.cols;
    int num_cels = (I0.rows - 1) * (I0.cols - 1) * 2;
    spdlog::info("precision:{} rows:{}, cols:{}, num_nods:{} num_cels:{}", sizeof(Float), I0.rows, I0.cols, num_nods, num_cels);

    matd_t nods;
    mati_t cels;
    {
        nods.resize(2, num_nods);
        cels.resize(3, num_cels);
        int vidx = 0, fidx = 0;
        for (int i = 0; i < I0.rows; ++i)
        {
            for (int j = 0; j < I0.cols; ++j)
            {
                nods(0, vidx) = i;
                nods(1, vidx) = j;
                ++vidx;
            }
        }
        for (int i = 0; i < I0.rows - 1; ++i)
        {
            for (int j = 0; j < I0.cols - 1; ++j)
            {
                int v0 = i * I0.cols + j;
                int v1 = i * I0.cols + (j + 1);
                int v2 = (i + 1) * I0.cols + j;
                int v3 = (i + 1) * I0.cols + (j + 1);
                cels(0, fidx) = v0;
                cels(1, fidx) = v3;
                cels(2, fidx) = v2;
                ++fidx;
                cels(0, fidx) = v0;
                cels(1, fidx) = v1;
                cels(2, fidx) = v3;
                ++fidx;
            }
        }
    }

    std::shared_ptr<img_poisson> prb = std::make_shared<img_poisson>(cels, nods, I0.rows, I0.cols);

    vecd_t x;
    prb->RHS((Float *)I0.data, x);
    const int mod = prb->lambda > 10 ? 1 : 1.0 / (0.1 * prb->lambda);
    const int max_iter = prb->lambda >= 10 ? 10 : mod * 10;
    for (int iter = 0; iter <= max_iter; ++iter) // 0.053198s for explicit, 3.259856s for implicit
    {
        auto start = get_time();

        Eigen::SparseMatrix<Float> A;
        prb->LHS(x.data(), A);
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<Float>> chol(A); // in implicit schema, lambda can be set to 1.0
        x = chol.solve(x);

        printf("iter%d time=%f\n", iter, get_time() - start);
        if (iter % mod == 0)
        {
            saveImg(x, I0.rows, I0.cols, outdir + to_string(iter) + ".png");
        }
    }
    return 0;
}
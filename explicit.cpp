#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <boost/filesystem.hpp>

#include "util.h"
using namespace std;
namespace fs = boost::filesystem;

double lambda = 0.1;
const double K = 10;
const double K2 = 1 / K / K;
// Xt+1 = Xt + lambda * dt * L(Xt) eq8 in "implicit fairing"
int main()
{
    string outdir = "output/cv/lam" + to_string(lambda) + "/";

    cv::Mat img = cv::imread("data/shell.png", cv::IMREAD_GRAYSCALE);
    // cv::resize(img, img, cv::Size(1280, 800));
    // cv::imwrite(outdir + "img.png", img);

    if (!fs::exists(outdir))
        fs::create_directories(outdir);

    cv::Mat It;
    img.convertTo(It, CV_32FC1);
    for (int iter = 0; iter <= 1000; ++iter)
    {
        auto start = get_time();

        cv::Mat dx, dy;
        cv::Sobel(It, dx, CV_32F, 1, 0, 3); // gradient along x
        cv::Sobel(It, dy, CV_32F, 0, 1, 3); // gradient along y
        // cout << "dx\n"
        //      << dx << endl;
        // cout << "dy\n"
        //      << dy << endl;
        // cv::imshow("dx", dx);
        // cv::imshow("dy", dy);

        cv::Mat C = cv::Mat::zeros(It.size(), CV_32F); // diffusion coeff
        for (int i = 0; i < It.rows; ++i)
            for (int j = 0; j < It.cols; ++j)
            {
                float gx = dx.at<float>(i, j), gy = dy.at<float>(i, j);
                float c;
                if (i == 0 || i == It.rows - 1 || j == 0 || j == It.cols - 1)
                    c = 0; // no diffusion on boundary
                else
                {
                    // c = std::exp(-(gx * gx + gy * gy) * K2);
                    c = 1.0 / (1.0 + (gx * gx + gy * gy) * K2);
                }
                C.at<float>(i, j) = c;
            }
        // cv::imshow("C", C);

        cv::Mat LI = cv::Mat::zeros(It.size(), CV_32F); // laplace(I)
        float maxLI = 0, intLI = 0;                     // max(laplace(I)), integrate(laplace(I))
        for (int i = 0; i < It.rows; ++i)
        {
            for (int j = 0; j < It.cols; ++j)
            {
                float Ip = It.at<float>(i, j);
                float In = (i == 0) ? Ip : It.at<float>(i - 1, j);
                float Is = (i == It.rows - 1) ? Ip : It.at<float>(i + 1, j);
                float Ie = (j == It.cols - 1) ? Ip : It.at<float>(i, j + 1);
                float Iw = (j == 0) ? Ip : It.at<float>(i, j - 1);

                float Cp = C.at<float>(i, j);
                float Cn = (i == 0) ? 0 : C.at<float>(i - 1, j);
                float Cs = (i == It.rows - 1) ? 0 : C.at<float>(i + 1, j);
                float Ce = (j == It.cols - 1) ? 0 : C.at<float>(i, j + 1);
                float Cw = (j == 0) ? 0 : C.at<float>(i, j - 1);

                float ddI = (Cn + Cp) * (In - Ip) +
                            (Cs + Cp) * (Is - Ip) +
                            (Ce + Cp) * (Ie - Ip) +
                            (Cw + Cp) * (Iw - Ip); // divergence(I)

                // float ddI = (In - Ip) +
                //             (Is - Ip) +
                //             (Ie - Ip) +
                //             (Iw - Ip); // pure laplace blurring

                LI.at<float>(i, j) = ddI;
                if (fabs(ddI) > maxLI)
                    maxLI = fabs(ddI);
                intLI += fabs(ddI);
            }
        }

        // lambda = 100 / maxLI;
        // It += lambda * 0.25 * LI;
        It += lambda * LI;

        // 0.001933s for 1 iter 636x398
        printf("iter%d: lambda=%f, maxLI=%f, intLI=%f time=%f\n", iter, lambda, maxLI, intLI, get_time() - start);
        if (iter % 100 == 0)
        {
            // cv::imshow("output/LI" + to_string(iter) + ".png", LI);
            // cv::waitKey(-1);

            cv::Mat ItTmp;
            It.convertTo(ItTmp, CV_8U);
            cv::imwrite(outdir + to_string(iter) + ".png", ItTmp);
        }
    }

    return 0;
}
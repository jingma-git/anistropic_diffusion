#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

double lambda = 0.25;
const double K = 10;
const double K2 = 1 / K / K;

int main()
{
    cv::Mat img = cv::imread("data/house.png", cv::IMREAD_GRAYSCALE);
    cv::imwrite("output/img.png", img);
    cv::Mat It;
    img.convertTo(It, CV_32FC1);

    for (int iter = 0; iter <= 1000; ++iter)
    {
        cv::Mat dx, dy;
        cv::Sobel(It, dx, CV_32F, 1, 0, 3); // gradient along x
        cv::Sobel(It, dy, CV_32F, 0, 1, 3); // gradient along y
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
                            (Cw + Cp) * (Iw - Ip);

                LI.at<float>(i, j) = ddI;
                if (fabs(ddI) > maxLI)
                    maxLI = fabs(ddI);
                intLI += fabs(ddI);
            }
        }

        lambda = 100 / maxLI;
        printf("iter%d: lambda=%f, maxLI=%f, intLI=%f\n", iter, lambda, maxLI, intLI);
        It += lambda / 4. * LI;
        if (iter % 100 == 0)
        {
            // cv::Mat LITmp;
            // LI.convertTo(LITmp, CV_8U);
            cv::imshow("output/LI" + to_string(iter) + ".png", LI);
            cv::waitKey(-1);

            cv::Mat ItTmp;
            It.convertTo(ItTmp, CV_8U);
            cv::imwrite("output/It" + to_string(iter) + ".png", ItTmp);
        }
    }

    return 0;
}
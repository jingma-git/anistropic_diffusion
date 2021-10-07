#include "ichol.h"
#include <iostream>

using namespace ichol;
using namespace std;
using namespace Eigen;

void testSymbA()
{
    spmat_t A(3, 3);
    vector<Triplet<Float>> data;
    data.emplace_back(0, 0, 4);
    data.emplace_back(1, 1, 2);
    data.emplace_back(2, 2, 11);
    data.emplace_back(0, 1, -2);
    data.emplace_back(1, 0, -1);
    data.emplace_back(0, 2, 2);
    data.emplace_back(2, 0, 2);
    data.emplace_back(1, 2, 4);
    data.emplace_back(2, 1, 4);
    A.setFromTriplets(data.begin(), data.end());
    vecd_t G = A.diagonal();
    cout << G.transpose() << endl;
    G = G.cwiseSqrt().cwiseInverse();
    cout << G.transpose() << endl;

    auto PTR = A.outerIndexPtr();
    auto IND = A.innerIndexPtr();
    auto VAL = A.valuePtr();
    for (int j = 0; j < A.cols(); ++j)
    {
        for (int iter = PTR[j]; iter < PTR[j + 1]; ++iter)
        {
            int i = IND[iter];
            VAL[iter] = G[i] * G[j];
            printf("VAL[%d, %d]=%g\n", i, j, VAL[iter]);
        }
    }

    cout << A.toDense() << endl;
}

void testOrderCast()
{
    spmat_t A(3, 3);
    vector<Triplet<Float>> data;
    data.emplace_back(0, 0, 4);
    data.emplace_back(1, 1, 2);
    data.emplace_back(2, 2, 11);
    data.emplace_back(0, 1, -2);
    data.emplace_back(0, 2, 2);
    data.emplace_back(1, 2, 4);
    A.setFromTriplets(data.begin(), data.end());
    for (int j = 0; j < A.cols(); ++j)
    {
        for (int iter = A.outerIndexPtr()[j]; iter < A.outerIndexPtr()[j + 1]; ++iter)
        {
            int i = A.innerIndexPtr()[iter];
            cout << i << ", " << j << " " << A.valuePtr()[iter] << endl;
        }
    }
    spmatr_t B = A;
    cout << "B\n"
         << B.toDense() << endl;
    for (int j = 0; j < B.cols(); ++j)
    {
        for (int iter = B.outerIndexPtr()[j]; iter < B.outerIndexPtr()[j + 1]; ++iter)
        {
            int i = B.innerIndexPtr()[iter];
            cout << i << ", " << j << " " << B.valuePtr()[iter] << endl;
        }
    }
}
int main()
{
    // testSymbA();
    testOrderCast();
    return 0;
}
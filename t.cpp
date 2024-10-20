#include "filters.h"
#include "matrix.h"
#include "ppm.h"
#include <cmath>
#include <immintrin.h>
#include <iostream>

void GetWeights(const int n, double* weightsOut) {
    for (int i = 0; i <= n; ++i) {
        double x = static_cast<double>(i) * maxX /n;
        weightsOut[i] = exp(-x*x*pi);
    }
}

inline unsigned int GetRIdx(unsigned x, unsigned y, unsigned xSize) {
    return (y * xSize + x)*3;
};

inline unsigned int GetGIdx(unsigned x, unsigned y, unsigned xSize) {
    return (y * xSize + x)*3 + 1;
}

inline unsigned int GetBIdx(unsigned x, unsigned y, unsigned xSize) {
    return (y * xSize + x)*3 + 2;
}


void Blur(Matrix* m, std::shared_ptr<std::barrier<>> barrier, int radius, int startPos, int endPos)
{
    Matrix scratch(m->get_x_size(), m->get_y_size());
    double w[max_radius];
    GetWeights(radius, w);

    int ySize = m->get_y_size();
    int xSize = m->get_x_size();
    unsigned char* mPtr = m->GetData();
    unsigned char* sPtr = scratch.GetData();

    int simdEnd = (radius / 4) * 4;

    for (int x = startPos; x < endPos; ++x) {
        for (int y = 0; y < ySize; ++y) {
            // Load weight for the center pixel
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            rSum += mPtr[GetRIdx(x,y, xSize)] * w[0];
            gSum += mPtr[GetGIdx(x,y, xSize)] * w[0];
            bSum += mPtr[GetBIdx(x,y, xSize)] * w[0];
            wSum += w[0];

            alignas(32) double r_values[8];
            alignas(32) double g_values[8];
            alignas(32) double b_values[8];
            alignas(32) double w_values[8];

            // Loop through the kernel radius, optimized for multiples of 4
            for (int wi = 1; wi <= simdEnd; wi += 4) {
                __m256d weight = _mm256_set_pd(w[wi + 3], w[wi + 2], w[wi + 1], w[wi]);

                // Handle x - wi (left side)
                int x_left = x - (wi+3);
                if (x_left >= 0) {
                    __m256d r_left = _mm256_set_pd(mPtr[GetRIdx(x_left, y, xSize)], mPtr[GetRIdx(x_left+1, y, xSize)], mPtr[GetRIdx(x_left+2, y, xSize)], mPtr[GetRIdx(x_left+3, y, xSize)]);
                    __m256d g_left = _mm256_set_pd(mPtr[GetGIdx(x_left, y, xSize)], mPtr[GetGIdx(x_left+1, y, xSize)], mPtr[GetGIdx(x_left+2, y, xSize)], mPtr[GetGIdx(x_left+3, y, xSize)]);
                    __m256d b_left = _mm256_set_pd(mPtr[GetBIdx(x_left, y, xSize)], mPtr[GetBIdx(x_left+1, y, xSize)], mPtr[GetBIdx(x_left+2, y, xSize)], mPtr[GetBIdx(x_left+3, y, xSize)]);

                    // Store intermediate results, defer summation
                    _mm256_store_pd(&r_values[0], _mm256_mul_pd(r_left, weight));
                    _mm256_store_pd(&g_values[0], _mm256_mul_pd(g_left, weight));
                    _mm256_store_pd(&b_values[0], _mm256_mul_pd(b_left, weight));
                    _mm256_store_pd(&w_values[0], weight);
                }
                else if(x_left+3 >= 0)
                {
                    int index = 0;
                    int temp = wi;
                    x_left += 3;
                    while(x_left >= 0) {
                        r_values[index] = w[temp] * mPtr[GetRIdx(x_left, y, xSize)];
                        g_values[index] = w[temp] * mPtr[GetGIdx(x_left, y, xSize)];
                        b_values[index] = w[temp] * mPtr[GetBIdx(x_left, y, xSize)];
                        w_values[index] = w[temp];

                        x_left--;
                        temp++;
                        index++;
                    }
                }

                // Handle x + wi (right side)
                int x_right = x + wi+3;
                if (x_right < xSize) {
                    __m256d r_right = _mm256_set_pd(mPtr[GetRIdx(x_right, y, xSize)], mPtr[GetRIdx(x_right-1, y, xSize)], mPtr[GetRIdx(x_right-2, y, xSize)], mPtr[GetRIdx(x_right-3, y, xSize)]);
                    __m256d g_right = _mm256_set_pd(mPtr[GetGIdx(x_right, y, xSize)], mPtr[GetGIdx(x_right-1, y, xSize)], mPtr[GetGIdx(x_right-2, y, xSize)], mPtr[GetGIdx(x_right-3, y, xSize)]);
                    __m256d b_right = _mm256_set_pd(mPtr[GetBIdx(x_right, y, xSize)], mPtr[GetBIdx(x_right-1, y, xSize)], mPtr[GetBIdx(x_right-2, y, xSize)], mPtr[GetBIdx(x_right-3, y, xSize)]);

                    _mm256_store_pd(&r_values[4], _mm256_mul_pd(r_right, weight));
                    _mm256_store_pd(&g_values[4], _mm256_mul_pd(g_right, weight));
                    _mm256_store_pd(&b_values[4], _mm256_mul_pd(b_right, weight));
                    _mm256_store_pd(&w_values[4], weight);
                }
                else if(x_right-3 < xSize) {
                    x_right -= 3;
                    int temp = wi;
                    int index = 4;
                    while(x_right < xSize)
                    {
                        r_values[index] = w[temp] * mPtr[GetRIdx(x_right, y, xSize)];
                        g_values[index] = w[temp] * mPtr[GetGIdx(x_right, y, xSize)];;
                        b_values[index] = w[temp] * mPtr[GetBIdx(x_right, y, xSize)];;
                        w_values[index] = w[temp];

                        index++;
                        x_right++;
                        temp++;
                    }
                }

                int idx = 0;
                for(int i = wi; i < wi+4; ++i) {
                    int x_left = x - i;

                    if(x_left >= 0) {
                        rSum += r_values[idx];
                        gSum += g_values[idx];
                        bSum += b_values[idx];
                        wSum += w_values[idx];
                    }

                    int x_right = x + i;
                    if(x_right < xSize) {
                        rSum += r_values[idx+4];
                        gSum += g_values[idx+4];
                        bSum += b_values[idx+4];
                        wSum += w_values[idx+4];
                    }

                    idx++;
                }
            }

            // Handle remaining weights (if any)
            for (int wi = simdEnd + 1; wi <= radius; ++wi) {
                double wc = w[wi];

                // Handle x - wi (left side)
                int x_left = x - wi;
                if (x_left >= 0) {
                    rSum += wc * mPtr[GetRIdx(x_left, y, xSize)];
                    gSum += wc * mPtr[GetGIdx(x_left, y, xSize)];
                    bSum += wc * mPtr[GetBIdx(x_left, y, xSize)];
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int x_right = x + wi;
                if (x_right < xSize) {
                    rSum += wc * mPtr[GetRIdx(x_right, y, xSize)];
                    gSum += wc * mPtr[GetGIdx(x_right, y, xSize)];
                    bSum += wc * mPtr[GetBIdx(x_right, y, xSize)];
                    wSum += wc;
                }
            }

            sPtr[GetRIdx(x, y, xSize)] = rSum/wSum;
            sPtr[GetGIdx(x, y, xSize)] = gSum/wSum;
            sPtr[GetBIdx(x, y, xSize)] = bSum/wSum;
        }
    }

    for(int x = 0; x < xSize; ++x) {
        for(int y = 0; y < ySize; ++y) {
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            rSum += sPtr[GetRIdx(x, y, xSize)] * w[0];
            gSum += sPtr[GetGIdx(x, y, xSize)] * w[0];
            bSum += sPtr[GetBIdx(x, y, xSize)] * w[0];
            wSum += w[0];

            alignas(32) double r_values[8];
            alignas(32) double g_values[8];
            alignas(32) double b_values[8];
            alignas(32) double w_values[8];

            for (int wi = 1; wi <= simdEnd; wi += 4) {
                __m256d weight = _mm256_set_pd(w[wi + 3], w[wi + 2], w[wi + 1], w[wi]);

                // Handle x - wi (left side)
                int y_left = y - (wi+3);
                if (y_left >= 0) {
                    __m256d r_left = _mm256_set_pd(sPtr[GetRIdx(x, y_left, xSize)], sPtr[GetRIdx(x, y_left+1, xSize)], sPtr[GetRIdx(x, y_left+2, xSize)], sPtr[GetRIdx(x, y_left+3, xSize)]);
                    __m256d g_left = _mm256_set_pd(sPtr[GetGIdx(x, y_left, xSize)], sPtr[GetGIdx(x, y_left+1, xSize)], sPtr[GetGIdx(x, y_left+2, xSize)], sPtr[GetGIdx(x, y_left+3, xSize)]);
                    __m256d b_left = _mm256_set_pd(sPtr[GetBIdx(x, y_left, xSize)], sPtr[GetBIdx(x, y_left+1, xSize)], sPtr[GetBIdx(x, y_left+2, xSize)], sPtr[GetBIdx(x, y_left+3, xSize)]);

                    _mm256_store_pd(&r_values[0], _mm256_mul_pd(r_left, weight));
                    _mm256_store_pd(&g_values[0], _mm256_mul_pd(g_left, weight));
                    _mm256_store_pd(&b_values[0], _mm256_mul_pd(b_left, weight));
                    _mm256_store_pd(&w_values[0], weight);
                }
                else if(y_left+3 >= 0)
                {
                    int index = 0;
                    int temp = wi;
                    y_left += 3;
                    while(y_left >= 0)
                    {
                        r_values[index] = w[temp] * sPtr[GetRIdx(x, y_left, xSize)];
                        g_values[index] = w[temp] * sPtr[GetGIdx(x, y_left, xSize)];
                        b_values[index] = w[temp] * sPtr[GetBIdx(x, y_left, xSize)];

                        w_values[index] = w[temp];

                        y_left--;
                        temp++;
                        index++;
                    }
                }

                // Handle x + wi (right side)
                int y_right = y + wi+3;
                if (y_right < ySize) {
                    __m256d r_right = _mm256_set_pd(sPtr[GetRIdx(x, y_right, xSize)], sPtr[GetRIdx(x, y_right-1, xSize)], sPtr[GetRIdx(x, y_right-2, xSize)], sPtr[GetRIdx(x, y_right-3, xSize)]);
                    __m256d g_right = _mm256_set_pd(sPtr[GetGIdx(x, y_right, xSize)], sPtr[GetGIdx(x, y_right-1, xSize)], sPtr[GetGIdx(x, y_right-2, xSize)], sPtr[GetGIdx(x, y_right-3, xSize)]);
                    __m256d b_right = _mm256_set_pd(sPtr[GetBIdx(x, y_right, xSize)], sPtr[GetBIdx(x, y_right-1, xSize)], sPtr[GetBIdx(x, y_right-2, xSize)], sPtr[GetBIdx(x, y_right-3, xSize)]);

                    _mm256_store_pd(&r_values[4], _mm256_mul_pd(r_right, weight));
                    _mm256_store_pd(&g_values[4], _mm256_mul_pd(g_right, weight));
                    _mm256_store_pd(&b_values[4], _mm256_mul_pd(b_right, weight));
                    _mm256_store_pd(&w_values[4], weight);
                }
                else if(y_right-3 < ySize)
                {
                    int index = 4;
                    y_right -= 3;
                    int temp = wi;
                    while(y_right < ySize)
                    {
                        r_values[index] = w[temp] * sPtr[GetRIdx(x, y_right, xSize)];
                        g_values[index] = w[temp] * sPtr[GetGIdx(x, y_right, xSize)];
                        b_values[index] = w[temp] * sPtr[GetBIdx(x, y_right, xSize)];
                        w_values[index] = w[temp];

                        y_right++;
                        temp++;
                        index++;
                    }
                }
                int idx = 0;
                for(int i = wi; i < wi+4; ++i) {
                    int y_left = y - i;

                    if(y_left >= 0) {
                        rSum += r_values[idx];
                        gSum += g_values[idx];
                        bSum += b_values[idx];
                        wSum += w_values[idx];
                    }

                    int y_right = y + i;
                    if(y_right < ySize) {
                        rSum += r_values[idx+4];
                        gSum += g_values[idx+4];
                        bSum += b_values[idx+4];
                        wSum += w_values[idx+4];
                    }

                    idx++;
                }
            }

            for (int wi = simdEnd + 1; wi <= radius; ++wi) {
                double wc = w[wi];

                // Handle x - wi (left side)
                int y_left = y - wi;
                if (y_left >= 0) {
                    rSum += wc * sPtr[GetRIdx(x, y_left, xSize)];
                    gSum += wc * sPtr[GetGIdx(x, y_left, xSize)];;
                    bSum += wc * sPtr[GetBIdx(x, y_left, xSize)];
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int y_right = y + wi;
                if (y_right < ySize) {
                    rSum += wc * sPtr[GetRIdx(x, y_right, xSize)];
                    gSum += wc * sPtr[GetGIdx(x, y_right, xSize)];
                    bSum += wc * sPtr[GetBIdx(x, y_right, xSize)];
                    wSum += wc;
                }
            }

            mPtr[GetRIdx(x, y, xSize)] = rSum/wSum;
            mPtr[GetGIdx(x, y, xSize)] = gSum/wSum;
            mPtr[GetBIdx(x, y, xSize)] = bSum/wSum;
        }
    }
}

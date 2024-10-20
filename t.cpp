#include "filters.h"
#include "matrix.h"
#include "ppm.h"
#include <cmath>
#include <immintrin.h>

inline unsigned GetBaseIdx(unsigned x, unsigned y, unsigned xSize)
{
    return y * xSize + x;
}

void GetWeights(const int n, double* weightsOut) {
    for (int i = 0; i <= n; ++i) {
        double x = static_cast<double>(i) * maxX /n;
        weightsOut[i] = exp(-x*x*pi);
    }
}

void Blur(Matrix* m, std::shared_ptr<std::barrier<>> barrier, int radius, int startPos, int endPos)
{
    Matrix scratch(m->get_x_size(), m->get_y_size());
    double w[max_radius];
    GetWeights(radius, w);

    int ySize = m->get_y_size();
    int xSize = m->get_x_size();

    unsigned char* R = m->get_R();
    unsigned char* G = m->get_G();
    unsigned char* B = m->get_B();

    int simdEnd = (radius / 4) * 4;

    for (int x = startPos; x < endPos; ++x) {
        for (int y = 0; y < ySize; ++y) {
            // Load weight for the center pixel
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            unsigned int baseIdx = GetBaseIdx(x, y, xSize);

            rSum += R[baseIdx] * w[0];
            gSum += G[baseIdx] * w[0];
            bSum += B[baseIdx] * w[0];
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
                if (x_left >= 0)
                {
                    baseIdx = GetBaseIdx(x_left, y, xSize);
                    __m128i chars = _mm_loadu_si128((__m128i*)&R[baseIdx]);
                    __m128i int32_vals = _mm_cvtepu8_epi32(chars);
                    __m256d double_vals = _mm256_cvtepi32_pd(int32_vals);

                    __m256d r_left = _mm256_set_pd(R[baseIdx], R[baseIdx+1], R[baseIdx+2], R[baseIdx+3]);
                    __m256d g_left = _mm256_set_pd(G[baseIdx], G[baseIdx+1], G[baseIdx+2], G[baseIdx+3]);
                    __m256d b_left = _mm256_set_pd(B[baseIdx], B[baseIdx+1], B[baseIdx+2], B[baseIdx+3]);

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
                        baseIdx = GetBaseIdx(x_left, y, xSize);
                        r_values[index] = w[temp] * R[baseIdx];
                        g_values[index] = w[temp] * G[baseIdx];
                        b_values[index] = w[temp] * B[baseIdx];
                        w_values[index] = w[temp];

                        x_left--;
                        temp++;
                        index++;
                    }
                }

                // Handle x + wi (right side)
                int x_right = x + wi+3;
                if (x_right < xSize) {
                    baseIdx = GetBaseIdx(x_left, y, xSize);
                    __m256d r_right = _mm256_set_pd(R[baseIdx], R[baseIdx-1], R[baseIdx-2], R[baseIdx-3]);
                    __m256d g_right = _mm256_set_pd(G[baseIdx], G[baseIdx-1], G[baseIdx-2], G[baseIdx-3]);
                    __m256d b_right = _mm256_set_pd(B[baseIdx], B[baseIdx-1], B[baseIdx-2], B[baseIdx-3]);

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
                        baseIdx = GetBaseIdx(x_right, y, xSize);
                        r_values[index] = w[temp] * R[baseIdx];
                        g_values[index] = w[temp] * G[baseIdx];
                        b_values[index] = w[temp] * B[baseIdx];
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
                    baseIdx = GetBaseIdx(x_left, y, xSize);
                    rSum += wc * R[baseIdx];
                    gSum += wc * G[baseIdx];
                    bSum += wc * B[baseIdx];
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int x_right = x + wi;
                if (x_right < xSize) {
                    baseIdx = GetBaseIdx(x_right, y, xSize);
                    rSum += wc * R[baseIdx];
                    gSum += wc * G[baseIdx];
                    bSum += wc * B[baseIdx];
                    wSum += wc;
                }
            }

            baseIdx = GetBaseIdx(x, y, xSize);
            scratch.get_R()[baseIdx] = rSum/wSum;
            scratch.get_G()[baseIdx] = gSum/wSum;
            scratch.get_B()[baseIdx] = bSum/wSum;
        }
    }

    barrier->arrive_and_wait();

    R = scratch.get_R();
    G = scratch.get_G();
    B = scratch.get_B();

    // Horizontal blur
    for(int x = 0; x < xSize; ++x) {
        for(int y = 0; y < ySize; ++y) {
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            unsigned baseIdx = GetBaseIdx(x, y, xSize);

            rSum += R[baseIdx] * w[0];
            gSum += G[baseIdx] * w[0];
            bSum += B[baseIdx] * w[0];
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
                    baseIdx = GetBaseIdx(x, y_left, xSize);
                    __m256d r_left = _mm256_set_pd(R[baseIdx], R[baseIdx+xSize], R[baseIdx+2*xSize], R[baseIdx+3*xSize]);
                    __m256d g_left = _mm256_set_pd(G[baseIdx], G[baseIdx+xSize], G[baseIdx+2*xSize], G[baseIdx+3*xSize]);
                    __m256d b_left = _mm256_set_pd(B[baseIdx], B[baseIdx+xSize], B[baseIdx+2*xSize], B[baseIdx+3*xSize]);

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
                        baseIdx = GetBaseIdx(x, y_left, xSize);
                        r_values[index] = w[temp] * R[baseIdx];
                        g_values[index] = w[temp] * G[baseIdx];
                        b_values[index] = w[temp] * B[baseIdx];

                        w_values[index] = w[temp];

                        y_left--;
                        temp++;
                        index++;
                    }
                }

                // Handle x + wi (right side)
                int y_right = y + wi+3;
                if (y_right < ySize) {
                    baseIdx = GetBaseIdx(x, y_right, xSize);
                    __m256d r_right = _mm256_set_pd(R[baseIdx], R[baseIdx-xSize], R[baseIdx-2*xSize], R[baseIdx-3*xSize]);
                    __m256d g_right = _mm256_set_pd(G[baseIdx], G[baseIdx-xSize], G[baseIdx-2*xSize], G[baseIdx-3*xSize]);
                    __m256d b_right = _mm256_set_pd(B[baseIdx], B[baseIdx-xSize], B[baseIdx-2*xSize], B[baseIdx-3*xSize]);

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
                        baseIdx = GetBaseIdx(x, y_right, xSize);
                        r_values[index] = w[temp] * R[baseIdx];
                        g_values[index] = w[temp] * G[baseIdx];
                        b_values[index] = w[temp] * B[baseIdx];
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
                    baseIdx = GetBaseIdx(x, y_left, xSize);
                    rSum += wc * R[baseIdx];
                    gSum += wc * G[baseIdx];
                    bSum += wc * B[baseIdx];
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int y_right = y + wi;
                if (y_right < ySize) {
                    baseIdx = GetBaseIdx(x, y_right, xSize);
                    rSum += wc * R[baseIdx];
                    gSum += wc * G[baseIdx];
                    bSum += wc * B[baseIdx];
                    wSum += wc;
                }
            }
            
            baseIdx = GetBaseIdx(x, y, xSize);
            m->get_R()[baseIdx] = rSum/wSum;
            m->get_G()[baseIdx] = gSum/wSum;
            m->get_B()[baseIdx] = bSum/wSum;
        }
    }
}

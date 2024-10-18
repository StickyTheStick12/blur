#include "filters.h"
#include "matrix.h"
#include "ppm.h"
#include <cmath>
#include <immintrin.h>

unsigned int GetRIdx(unsigned x, unsigned y, unsigned xSize) {
    return (y * xSize + x)*3;
};

unsigned int GetGIdx(unsigned x, unsigned y, unsigned xSize) {
    return (y * xSize + x)*3 + 1;
}

unsigned int GetBIdx(unsigned x, unsigned y, unsigned xSize) {
    return (y * xSize + x)*3 + 2;
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
    unsigned char* mPtr = m->GetData();
    unsigned char* sPtr = scratch.GetData();

    for (int x = startPos; x < endPos; ++x) {
        for (int y = 0; y < ySize; ++y) {
            // Load weight for the center pixel
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            __m256d sum_w = _mm256_setzero_pd();
            __m256d w0 = _mm256_set_pd(w[0], 0, 0, 0);

            // Load the pixel values at (x, y) for R, G, B
            __m256d r = _mm256_set_pd(mPtr[GetRIdx(x,y, xSize)], 0, 0, 0);
            __m256d g = _mm256_set_pd(mPtr[GetGIdx(x,y, xSize)], 0, 0, 0);
            __m256d b = _mm256_set_pd(mPtr[GetBIdx(x,y, xSize)], 0, 0, 0);

            // Apply weight for the center pixel
            __m256d sum_r = _mm256_mul_pd(r, w0);
            __m256d sum_g = _mm256_mul_pd(g, w0);
            __m256d sum_b = _mm256_mul_pd(b, w0);
            sum_w = _mm256_add_pd(sum_w, w0);

            // Loop through the kernel radius, optimized for multiples of 4
            for (int wi = 1; wi <= radius; wi += 4) {
                __m256d weight = _mm256_set_pd(w[wi + 3], w[wi + 2], w[wi + 1], w[wi]);

                // Handle x - wi (left side)
                int x_left = x - (wi+3);
                if (x_left >= 0) {
                    __m256d r_left = _mm256_set_pd(mPtr[GetRIdx(x_left+3, y, xSize)], mPtr[GetRIdx(x_left+2, y, xSize)], mPtr[GetRIdx(x_left+1, y, xSize)], mPtr[GetRIdx(x_left, y, xSize)]);
                    __m256d g_left = _mm256_set_pd(mPtr[GetGIdx(x_left+3, y, xSize)], mPtr[GetGIdx(x_left+2, y, xSize)], mPtr[GetGIdx(x_left+1, y, xSize)], mPtr[GetGIdx(x_left, y, xSize)]);
                    __m256d b_left = _mm256_set_pd(mPtr[GetBIdx(x_left+3, y, xSize)], mPtr[GetBIdx(x_left+2, y, xSize)], mPtr[GetBIdx(x_left+1, y, xSize)], mPtr[GetBIdx(x_left, y, xSize)]);

                    sum_r = _mm256_fmadd_pd(r_left, weight, sum_r);
                    sum_g = _mm256_fmadd_pd(g_left, weight, sum_g);
                    sum_b = _mm256_fmadd_pd(b_left, weight, sum_b);

                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(x_left+3 >= 0)
                {
                    int temp = wi;
                    x_left += 3;
                    while(x_left >= 0)
                    {
                        rSum += w[temp] * mPtr[GetRIdx(x_left, y, xSize)];
                        gSum += w[temp] * mPtr[GetGIdx(x_left, y, xSize)];
                        bSum += w[temp] * mPtr[GetBIdx(x_left, y, xSize)];
                        wSum += w[temp];

                        x_left--;
                        temp--;
                    }
                }

                // Handle x + wi (right side)
                int x_right = x + wi+3;
                if (x_right < xSize) {
                    __m256d r_right = _mm256_set_pd(mPtr[GetRIdx(x_right, y, xSize)], mPtr[GetRIdx(x_right-1, y, xSize)], mPtr[GetRIdx(x_right-2, y, xSize)], mPtr[GetRIdx(x_right-3, y, xSize)]);
                    __m256d g_right = _mm256_set_pd(mPtr[GetGIdx(x_right, y, xSize)], mPtr[GetGIdx(x_right-1, y, xSize)], mPtr[GetGIdx(x_right-2, y, xSize)], mPtr[GetGIdx(x_right-3, y, xSize)]);
                    __m256d b_right = _mm256_set_pd(mPtr[GetBIdx(x_right, y, xSize)], mPtr[GetBIdx(x_right-1, y, xSize)], mPtr[GetBIdx(x_right-2, y, xSize)], mPtr[GetBIdx(x_right-3, y, xSize)]);

                    sum_r = _mm256_fmadd_pd(r_right, weight, sum_r); // sum_r = r_left * weight + sum_r
                    sum_g = _mm256_fmadd_pd(g_right, weight, sum_g); // sum_g = g_left * weight + sum_g
                    sum_b = _mm256_fmadd_pd(b_right, weight, sum_b); // sum_b = b_left * weight + sum_b

                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(x_right-3 < xSize)
                {
                    x_right -= 3;
                    int temp = wi;
                    while(x_right < xSize)
                    {
                        rSum += w[temp] * mPtr[GetRIdx(x_right, y, xSize)];
                        gSum += w[temp] * mPtr[GetGIdx(x_right, y, xSize)];
                        bSum += w[temp] * mPtr[GetBIdx(x_right, y, xSize)];
                        wSum += w[temp];

                        x_right++;
                        temp++;
                    }
                }
            }

            // Handle remaining weights (if any)
            for (int wi = radius - (radius % 4) + 2; wi <= radius; ++wi) {
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

            alignas(32) double results_r[4], results_g[4], results_b[4], result_w[4];
            _mm256_store_pd(results_r, sum_r);
            _mm256_store_pd(results_g, sum_g);
            _mm256_store_pd(results_b, sum_b);
            _mm256_store_pd(result_w, sum_w);

            rSum += results_r[0]+ results_r[1] + results_r[2] + results_r[3];
            gSum += results_g[0] + results_g[1] + results_g[2] + results_g[3];
            bSum += results_b[0] + results_b[1] + results_b[2] + results_b[3];
            wSum += result_w[0] + result_w[1] + result_w[2] + result_w[3];

            // Clamp the values
            sPtr[GetRIdx(x, y, xSize)] = rSum/wSum;
            sPtr[GetGIdx(x, y, xSize)] = gSum/wSum;
            sPtr[GetBIdx(x, y, xSize)] = bSum/wSum;
        }
    }

    barrier->arrive_and_wait();

    // Horizontal blur
    for(int x = startPos; x < endPos; ++x) {
        for(int y = 0; y < ySize; ++y) {
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            __m256d sum_w = _mm256_setzero_pd();
            __m256d w0 = _mm256_set_pd(w[0], 0, 0, 0);

            // Load the pixel values at (x, y) for R, G, B
            __m256d r = _mm256_set_pd(sPtr[GetRIdx(x,y, xSize)], 0, 0, 0);
            __m256d g = _mm256_set_pd(sPtr[GetGIdx(x,y, xSize)], 0, 0, 0);
            __m256d b = _mm256_set_pd(sPtr[GetBIdx(x,y, xSize)], 0, 0, 0);

            // Apply weight for the center pixel
            __m256d sum_r = _mm256_mul_pd(r, w0);
            __m256d sum_g = _mm256_mul_pd(g, w0);
            __m256d sum_b = _mm256_mul_pd(b, w0);
            sum_w = _mm256_add_pd(sum_w, w0);

            for (int wi = 1; wi <= radius; wi += 4) {
                __m256d weight = _mm256_set_pd(w[wi + 3], w[wi + 2], w[wi + 1], w[wi]);

                // Handle x - wi (left side)
                int y_left = y - (wi+3);
                if (y_left >= 0) {
                    __m256d r_left = _mm256_set_pd(sPtr[GetRIdx(x, y_left+3, xSize)], sPtr[GetRIdx(x, y_left+2, xSize)], sPtr[GetRIdx(x, y_left+1, xSize)], sPtr[GetRIdx(x, y_left, xSize)]);
                    __m256d g_left = _mm256_set_pd(sPtr[GetGIdx(x, y_left+3, xSize)], sPtr[GetGIdx(x, y_left+2, xSize)], sPtr[GetGIdx(x, y_left+1, xSize)], sPtr[GetGIdx(x, y_left, xSize)]);
                    __m256d b_left = _mm256_set_pd(sPtr[GetBIdx(x, y_left+3, xSize)], sPtr[GetBIdx(x, y_left+2, xSize)], sPtr[GetBIdx(x, y_left+1, xSize)], sPtr[GetBIdx(x, y_left, xSize)]);

                    sum_r = _mm256_fmadd_pd(r_left, weight, sum_r); // sum_r = r_left * weight + sum_r
                    sum_g = _mm256_fmadd_pd(g_left, weight, sum_g); // sum_g = g_left * weight + sum_g
                    sum_b = _mm256_fmadd_pd(b_left, weight, sum_b); // sum_b = b_left * weight + sum_b

                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(y_left+3 >= 0)
                {
                    int temp = wi;
                    y_left += 3;
                    while(y_left >= 0)
                    {
                        rSum += w[temp] * sPtr[GetRIdx(x, y_left, xSize)];
                        gSum += w[temp] * sPtr[GetGIdx(x, y_left, xSize)];
                        bSum += w[temp] * sPtr[GetBIdx(x, y_left, xSize)];

                        wSum += w[temp];

                        y_left--;
                        temp--;
                    }
                }

                // Handle x + wi (right side)
                int y_right = y + wi+3;
                if (y_right < ySize) {
                    __m256d r_right = _mm256_set_pd(sPtr[GetRIdx(x, y_right, xSize)], sPtr[GetRIdx(x, y_right-1, xSize)], sPtr[GetRIdx(x, y_right-2, xSize)], sPtr[GetRIdx(x, y_right-3, xSize)]);
                    __m256d g_right = _mm256_set_pd(sPtr[GetGIdx(x, y_right, xSize)], sPtr[GetGIdx(x, y_right-1, xSize)], sPtr[GetGIdx(x, y_right-2, xSize)], sPtr[GetGIdx(x, y_right-3, xSize)]);
                    __m256d b_right = _mm256_set_pd(sPtr[GetBIdx(x, y_right, xSize)], sPtr[GetBIdx(x, y_right-1, xSize)], sPtr[GetBIdx(x, y_right-2, xSize)], sPtr[GetBIdx(x, y_right-3, xSize)]);

                    sum_r = _mm256_fmadd_pd(r_right, weight, sum_r); // sum_r = r_left * weight + sum_r
                    sum_g = _mm256_fmadd_pd(g_right, weight, sum_g); // sum_g = g_left * weight + sum_g
                    sum_b = _mm256_fmadd_pd(b_right, weight, sum_b); // sum_b = b_left * weight + sum_b

                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(y_right-3 < ySize)
                {
                    y_right -= 3;
                    int temp = wi;
                    while(y_right < ySize)
                    {
                        rSum += w[temp] * sPtr[GetRIdx(x, y_right, xSize)];
                        gSum += w[temp] * sPtr[GetGIdx(x, y_right, xSize)];
                        bSum += w[temp] * sPtr[GetBIdx(x, y_right, xSize)];
                        wSum += w[temp];

                        y_right++;
                        temp++;
                    }
                }
            }

            for (int wi = radius - (radius % 4) + 2; wi <= radius; ++wi) {
                double wc = w[wi];

                // Handle x - wi (left side)
                int y_left = y - wi;
                if (y_left >= 0) {
                    rSum += wc * sPtr[GetRIdx(x, y_left, xSize)];
                    gSum += wc * sPtr[GetGIdx(x, y_left, xSize)];
                    bSum += wc * sPtr[GetBIdx(x, y_left, xSize)];
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int y_right = x + wi;
                if (y_right < ySize) {
                    rSum += wc * sPtr[GetRIdx(x, y_right, xSize)];
                    gSum += wc * sPtr[GetGIdx(x, y_right, xSize)];
                    bSum += wc * sPtr[GetBIdx(x, y_right, xSize)];
                    wSum += wc;
                }
            }

            alignas(32) double results_r[4], results_g[4], results_b[4], result_w[4];
            _mm256_store_pd(results_r, sum_r);
            _mm256_store_pd(results_g, sum_g);
            _mm256_store_pd(results_b, sum_b);
            _mm256_store_pd(result_w, sum_w);

            rSum += results_r[0] + results_r[1] + results_r[2] + results_r[3];
            gSum += results_g[0] + results_g[1] + results_g[2] + results_g[3];
            bSum += results_b[0] + results_b[1] + results_b[2] + results_b[3];
            wSum += result_w[0] + result_w[1] + result_w[2] + result_w[3];

            mPtr[GetRIdx(x, y, xSize)] = rSum/wSum;
            mPtr[GetGIdx(x, y, xSize)] = gSum/wSum;
            mPtr[GetBIdx(x, y, xSize)] = bSum/wSum;
        }
    }
}

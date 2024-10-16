#include "filters.h"
#include "matrix.h"
#include "ppm.h"
#include <cmath>
#include <immintrin.h>

void GetWeights(const int n, double* weightsOut) {
    for (int i = 0; i <= n; ++i) {
        double x = static_cast<double>(i) * maxX /n;
        weightsOut[i] = exp(-x*x*pi);
    }
}

void Blur(std::shared_ptr<Matrix> m, std::shared_ptr<std::barrier<>> barrier, int radius, int startPos, int endPos)
{
    Matrix scratch(3000);  // Assuming a temp buffer size of 3000, replace this with your dynamic size
    double w[max_radius];
    GetWeights(radius, w);

    int ySize = m->get_y_size();

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
            __m256d r = _mm256_set_pd(m->r(x, y), 0, 0, 0);
            __m256d g = _mm256_set_pd(m->g(x, y), 0, 0, 0);
            __m256d b = _mm256_set_pd(m->b(x, y), 0, 0, 0);

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
                    __m256d r_left = _mm256_set_pd(m->r(x_left+3, y), m->r(x_left+2, y), m->r(x_left+1, y), m->r(x_left, y));
                    __m256d g_left = _mm256_set_pd(m->g(x_left+3, y), m->g(x_left+2, y), m->g(x_left+1, y), m->g(x_left, y));
                    __m256d b_left = _mm256_set_pd(m->b(x_left+3, y), m->b(x_left+2, y), m->b(x_left+1, y), m->b(x_left, y));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_left, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_left, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_left, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(x_left+3 >= 0)
                {
                    int temp = wi;
                    x_left += 3;
                    while(x_left >= 0)
                    {
                        rSum += w[temp] * m->r(x_left, y);
                        gSum += w[temp] * m->g(x_left, y);
                        bSum += w[temp] * m->b(x_left, y);
                        wSum += w[temp];

                        x_left--;
                        temp--;
                    }
                }

                // Handle x + wi (right side)
                int x_right = x + wi+3;
                if (x_right < m->get_x_size()) {
                    __m256d r_right = _mm256_set_pd(m->r(x_right, y), m->r(x_right-1, y), m->r(x_right-2, y), m->r(x_right-3, y));
                    __m256d g_right = _mm256_set_pd(m->g(x_right, y), m->g(x_right-1, y), m->g(x_right-2, y), m->g(x_right-3, y));
                    __m256d b_right = _mm256_set_pd(m->b(x_right, y), m->b(x_right-1, y), m->b(x_right-2, y), m->b(x_right-3, y));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_right, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_right, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_right, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(x_right-3 < m->get_x_size())
                {
                    x_right -= 3;
                    int temp = wi;
                    while(x_right < m->get_x_size())
                    {
                        rSum += w[temp] * m->r(x_right, y);
                        gSum += w[temp] * m->g(x_right, y);
                        bSum += w[temp] * m->b(x_right, y);
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
                    rSum += wc * m->r(x_left, y);
                    gSum += wc * m->g(x_left, y);
                    bSum += wc * m->b(x_left, y);
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int x_right = x + wi;
                if (x_right < m->get_x_size()) {
                    rSum += wc * m->r(x_right, y);
                    gSum += wc * m->g(x_right, y);
                    bSum += wc * m->b(x_right, y);
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
            scratch.r(x, y) = rSum/wSum;
            scratch.g(x, y) = gSum/wSum;
            scratch.b(x, y) = bSum/wSum;
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
            __m256d r = _mm256_set_pd(scratch.r(x, y), 0, 0, 0);
            __m256d g = _mm256_set_pd(scratch.g(x, y), 0, 0, 0);
            __m256d b = _mm256_set_pd(scratch.b(x, y), 0, 0, 0);

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
                    __m256d r_left = _mm256_set_pd(scratch.r(x, y_left+3), scratch.r(x, y_left+2), scratch.r(x, y_left+1), scratch.r(x, y_left));
                    __m256d g_left = _mm256_set_pd(scratch.g(x, y_left+3), scratch.g(x, y_left+2), scratch.g(x, y_left+1), scratch.g(x, y_left));
                    __m256d b_left = _mm256_set_pd(scratch.b(x, y_left+3), scratch.b(x, y_left+2), scratch.b(x, y_left+1), scratch.b(x, y_left));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_left, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_left, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_left, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(y_left+3 >= 0)
                {
                    int temp = wi;
                    y_left += 3;
                    while(y_left >= 0)
                    {
                        rSum += w[temp] * scratch.r(x, y_left);
                        gSum += w[temp] * scratch.g(x, y_left);
                        bSum += w[temp] * scratch.b(x, y_left);
                        wSum += w[temp];

                        y_left--;
                        temp--;
                    }
                }

                // Handle x + wi (right side)
                int y_right = y + wi+3;
                if (y_right < ySize) {
                    __m256d r_right = _mm256_set_pd(scratch.r(x, y_right), scratch.r(x, y_right-1), scratch.r(x, y_right-2), scratch.r(x, y_right-3));
                    __m256d g_right = _mm256_set_pd(scratch.g(x, y_right), scratch.g(x, y_right-1), scratch.g(x, y_right-2), scratch.g(x, y_right-3));
                    __m256d b_right = _mm256_set_pd(scratch.b(x, y_right), scratch.b(x, y_right-1), scratch.b(x, y_right-2), scratch.b(x, y_right-3));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_right, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_right, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_right, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(y_right-3 < ySize)
                {
                    y_right -= 3;
                    int temp = wi;
                    while(y_right < ySize)
                    {
                        rSum += w[temp] * scratch.r(x, y_right);
                        gSum += w[temp] * scratch.g(x, y_right);
                        bSum += w[temp] * scratch.b(x, y_right);
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
                    rSum += wc * m->r(x, y_left);
                    gSum += wc * m->g(x, y_left);
                    bSum += wc * m->b(x, y_left);
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int y_right = x + wi;
                if (y_right < ySize) {
                    rSum += wc * m->r(x, y_right);
                    gSum += wc * m->g(x, y_right);
                    bSum += wc * m->b(x, y_right);
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

            m->r(x, y) = rSum/wSum;
            m->g(x, y) = gSum/wSum;
            m->b(x, y) = bSum/wSum;
        }
    }
}

void Blur(Matrix& m, const int radius)
{
    Matrix scratch(3000);
    double w[max_radius];
    GetWeights(radius, w);

    int xSize = m.get_x_size();
    int ySize = m.get_y_size();

    for (int x = 0; x < xSize; ++x) {
        for (int y = 0; y < ySize; ++y) {
            // Load weight for the center pixel
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            __m256d sum_w = _mm256_setzero_pd();
            __m256d w0 = _mm256_set_pd(w[0], 0, 0, 0);

            // Load the pixel values at (x, y) for R, G, B
            __m256d r = _mm256_set_pd(m.r(x, y), 0, 0, 0);
            __m256d g = _mm256_set_pd(m.g(x, y), 0, 0, 0);
            __m256d b = _mm256_set_pd(m.b(x, y), 0, 0, 0);

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
                    __m256d r_left = _mm256_set_pd(m.r(x_left+3, y), m.r(x_left+2, y), m.r(x_left+1, y), m.r(x_left, y));
                    __m256d g_left = _mm256_set_pd(m.g(x_left+3, y), m.g(x_left+2, y), m.g(x_left+1, y), m.g(x_left, y));
                    __m256d b_left = _mm256_set_pd(m.b(x_left+3, y), m.b(x_left+2, y), m.b(x_left+1, y), m.b(x_left, y));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_left, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_left, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_left, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(x_left+3 >= 0)
                {
                    int temp = wi;
                    x_left += 3;
                    while(x_left >= 0)
                    {
                        rSum += w[temp] * m.r(x_left, y);
                        gSum += w[temp] * m.g(x_left, y);
                        bSum += w[temp] * m.b(x_left, y);
                        wSum += w[temp];

                        x_left--;
                        temp--;
                    }
                }

                // Handle x + wi (right side)
                int x_right = x + wi+3;
                if (x_right < xSize) {
                    __m256d r_right = _mm256_set_pd(m.r(x_right, y), m.r(x_right-1, y), m.r(x_right-2, y), m.r(x_right-3, y));
                    __m256d g_right = _mm256_set_pd(m.g(x_right, y), m.g(x_right-1, y), m.g(x_right-2, y), m.g(x_right-3, y));
                    __m256d b_right = _mm256_set_pd(m.b(x_right, y), m.b(x_right-1, y), m.b(x_right-2, y), m.b(x_right-3, y));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_right, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_right, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_right, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(x_right-3 < xSize)
                {
                    x_right -= 3;
                    int temp = wi;
                    while(x_right < xSize)
                    {
                        rSum += w[temp] * m.r(x_right, y);
                        gSum += w[temp] * m.g(x_right, y);
                        bSum += w[temp] * m.b(x_right, y);
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
                    rSum += wc * m.r(x_left, y);
                    gSum += wc * m.g(x_left, y);
                    bSum += wc * m.b(x_left, y);
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int x_right = x + wi;
                if (x_right < xSize) {
                    rSum += wc * m.r(x_right, y);
                    gSum += wc * m.g(x_right, y);
                    bSum += wc * m.b(x_right, y);
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
            scratch.r(x, y) = rSum/wSum;
            scratch.g(x, y) = gSum/wSum;
            scratch.b(x, y) = bSum/wSum;
        }
    }

    for(int x = 0; x < xSize; ++x) {
        for(int y = 0; y < ySize; ++y) {
            double rSum = 0;
            double gSum = 0;
            double bSum = 0;
            double wSum = 0;

            __m256d sum_w = _mm256_setzero_pd();
            __m256d w0 = _mm256_set_pd(w[0], 0, 0, 0);

            // Load the pixel values at (x, y) for R, G, B
            __m256d r = _mm256_set_pd(scratch.r(x, y), 0, 0, 0);
            __m256d g = _mm256_set_pd(scratch.g(x, y), 0, 0, 0);
            __m256d b = _mm256_set_pd(scratch.b(x, y), 0, 0, 0);

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
                    __m256d r_left = _mm256_set_pd(scratch.r(x, y_left+3), scratch.r(x, y_left+2), scratch.r(x, y_left+1), scratch.r(x, y_left));
                    __m256d g_left = _mm256_set_pd(scratch.g(x, y_left+3), scratch.g(x, y_left+2), scratch.g(x, y_left+1), scratch.g(x, y_left));
                    __m256d b_left = _mm256_set_pd(scratch.b(x, y_left+3), scratch.b(x, y_left+2), scratch.b(x, y_left+1), scratch.b(x, y_left));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_left, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_left, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_left, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(y_left+3 >= 0)
                {
                    int temp = wi;
                    y_left += 3;
                    while(y_left >= 0)
                    {
                        rSum += w[temp] * scratch.r(x, y_left);
                        gSum += w[temp] * scratch.g(x, y_left);
                        bSum += w[temp] * scratch.b(x, y_left);
                        wSum += w[temp];

                        y_left--;
                        temp--;
                    }
                }

                // Handle x + wi (right side)
                int y_right = y + wi+3;
                if (y_right < ySize) {
                    __m256d r_right = _mm256_set_pd(scratch.r(x, y_right), scratch.r(x, y_right-1), scratch.r(x, y_right-2), scratch.r(x, y_right-3));
                    __m256d g_right = _mm256_set_pd(scratch.g(x, y_right), scratch.g(x, y_right-1), scratch.g(x, y_right-2), scratch.g(x, y_right-3));
                    __m256d b_right = _mm256_set_pd(scratch.b(x, y_right), scratch.b(x, y_right-1), scratch.b(x, y_right-2), scratch.b(x, y_right-3));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_right, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_right, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_right, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
                else if(y_right-3 < ySize)
                {
                    y_right -= 3;
                    int temp = wi;
                    while(y_right < ySize)
                    {
                        rSum += w[temp] * scratch.r(x, y_right);
                        gSum += w[temp] * scratch.g(x, y_right);
                        bSum += w[temp] * scratch.b(x, y_right);
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
                    rSum += wc * m.r(x, y_left);
                    gSum += wc * m.g(x, y_left);
                    bSum += wc * m.b(x, y_left);
                    wSum += wc;
                }

                // Handle x + wi (right side)
                int y_right = x + wi;
                if (y_right < ySize) {
                    rSum += wc * m.r(x, y_right);
                    gSum += wc * m.g(x, y_right);
                    bSum += wc * m.b(x, y_right);
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

            m.r(x, y) = rSum/wSum;
            m.g(x, y) = gSum/wSum;
            m.b(x, y) = bSum/wSum;
        }
    }
}
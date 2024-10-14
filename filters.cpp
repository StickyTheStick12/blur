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


Matrix Blur(Matrix& m, const int radius)
{
    Matrix scratch(3000);  // Assuming a temp buffer size of 3000, replace this with your dynamic size
    double w[radius];
    GetWeights(radius, w);

    int xSize = m.get_x_size();
    int ySize = m.get_y_size();

    for (int x = 0; x < xSize; ++x) {
        for (int y = 0; y < ySize; y += 4) {
            // Initialize sum vectors for R, G, B, and weight accumulation
            __m256d sum_r = _mm256_setzero_pd();
            __m256d sum_g = _mm256_setzero_pd();
            __m256d sum_b = _mm256_setzero_pd();
            __m256d sum_w = _mm256_setzero_pd();

            // Load weight for the center pixel
            __m256d w0 = _mm256_set1_pd(w[0]);

            // Load the pixel values at (x, y) for R, G, B
            __m256d r = _mm256_set_pd(m.r(x, y + 3), m.r(x, y + 2), m.r(x, y + 1), m.r(x, y));
            __m256d g = _mm256_set_pd(m.g(x, y + 3), m.g(x, y + 2), m.g(x, y + 1), m.g(x, y));
            __m256d b = _mm256_set_pd(m.b(x, y + 3), m.b(x, y + 2), m.b(x, y + 1), m.b(x, y));

            // Apply weight and accumulate
            sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r, w0));
            sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g, w0));
            sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b, w0));
            sum_w = _mm256_add_pd(sum_w, w0);

            // Loop through the kernel radius
            for (int wi = 1; wi <= radius; ++wi) {
                double wc = w[wi];
                __m256d weight = _mm256_set1_pd(wc);

                // Handle x - wi (left side)
                int x_left = x - wi;
                if (x_left >= 0) {
                    __m256d r_left = _mm256_set_pd(m.r(x_left, y + 3), m.r(x_left, y + 2), m.r(x_left, y + 1), m.r(x_left, y));
                    __m256d g_left = _mm256_set_pd(m.g(x_left, y + 3), m.g(x_left, y + 2), m.g(x_left, y + 1), m.g(x_left, y));
                    __m256d b_left = _mm256_set_pd(m.b(x_left, y + 3), m.b(x_left, y + 2), m.b(x_left, y + 1), m.b(x_left, y));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_left, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_left, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_left, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }

                // Handle x + wi (right side)
                int x_right = x + wi;
                if (x_right < xSize) {
                    __m256d r_right = _mm256_set_pd(m.r(x_right, y + 3), m.r(x_right, y + 2), m.r(x_right, y + 1), m.r(x_right, y));
                    __m256d g_right = _mm256_set_pd(m.g(x_right, y + 3), m.g(x_right, y + 2), m.g(x_right, y + 1), m.g(x_right, y));
                    __m256d b_right = _mm256_set_pd(m.b(x_right, y + 3), m.b(x_right, y + 2), m.b(x_right, y + 1), m.b(x_right, y));

                    sum_r = _mm256_add_pd(sum_r, _mm256_mul_pd(r_right, weight));
                    sum_g = _mm256_add_pd(sum_g, _mm256_mul_pd(g_right, weight));
                    sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(b_right, weight));
                    sum_w = _mm256_add_pd(sum_w, weight);
                }
            }

            // Normalize the sum by dividing by the total weight
            sum_r = _mm256_div_pd(sum_r, sum_w);
            sum_g = _mm256_div_pd(sum_g, sum_w);
            sum_b = _mm256_div_pd(sum_b, sum_w);

            // Store the results back in the scratch matrix
            _mm256_storeu_pd(reinterpret_cast<double *>(&scratch.r(x, y)), sum_r);
            _mm256_storeu_pd(reinterpret_cast<double *>(&scratch.g(x, y)), sum_g);
            _mm256_storeu_pd(reinterpret_cast<double *>(&scratch.b(x, y)), sum_b);
        }
    }

    // Horizontal blur
    for (int x = 0; x < xSize; ++x) {
        for (int y = 0; y < ySize; ++y) {
            double r = w[0] * scratch.r(x, y);
            double g = w[0] * scratch.g(x, y);
            double b = w[0] * scratch.b(x, y);
            double n = w[0];

            for (int wi = 1; wi <= radius; ++wi) {
                double wc = w[wi];

                // Handle y - wi
                int y_left = y - wi;
                if (y_left >= 0) {
                    r += wc * scratch.r(x, y_left);
                    g += wc * scratch.g(x, y_left);
                    b += wc * scratch.b(x, y_left);
                    n += wc;
                }

                // Handle y + wi
                int y_right = y + wi;
                if (y_right < ySize) {
                    r += wc * scratch.r(x, y_right);
                    g += wc * scratch.g(x, y_right);
                    b += wc * scratch.b(x, y_right);
                    n += wc;
                }
            }

            // Normalize and store the final blurred pixel back into the original matrix
            m.r(x, y) = r / n;
            m.g(x, y) = g / n;
            m.b(x, y) = b / n;
        }
    }

    return m;
}
Matrix blur(Matrix m, const int radius)
    {
        Matrix scratch{3000};
        auto dst{m};

        for (auto x{0}; x < dst.get_x_size(); x++)
        {
            for (auto y{0}; y < dst.get_y_size(); y++)
            {
                double w[max_radius]{};
                GetWeights(radius, w);

                auto r{w[0] * dst.r(x, y)}, g{w[0] * dst.g(x, y)}, b{w[0] * dst.b(x, y)}, n{w[0]};

                for (auto wi{1}; wi <= radius; wi++)
                {
                    auto wc{w[wi]};
                    auto x2{x - wi};
                    if (x2 >= 0)
                    {
                        r += wc * dst.r(x2, y);
                        g += wc * dst.g(x2, y);
                        b += wc * dst.b(x2, y);
                        n += wc;
                    }
                    x2 = x + wi;
                    if (x2 < dst.get_x_size())
                    {
                        r += wc * dst.r(x2, y);
                        g += wc * dst.g(x2, y);
                        b += wc * dst.b(x2, y);
                        n += wc;
                    }
                }
                scratch.r(x, y) = r / n;
                scratch.g(x, y) = g / n;
                scratch.b(x, y) = b / n;
            }
        }

        for (auto x{0}; x < dst.get_x_size(); x++)
        {
            for (auto y{0}; y < dst.get_y_size(); y++)
            {
                double w[max_radius]{};
                GetWeights(radius, w);

                auto r{w[0] * scratch.r(x, y)}, g{w[0] * scratch.g(x, y)}, b{w[0] * scratch.b(x, y)}, n{w[0]};

                for (auto wi{1}; wi <= radius; wi++)
                {
                    auto wc{w[wi]};
                    auto y2{y - wi};
                    if (y2 >= 0)
                    {
                        r += wc * scratch.r(x, y2);
                        g += wc * scratch.g(x, y2);
                        b += wc * scratch.b(x, y2);
                        n += wc;
                    }
                    y2 = y + wi;
                    if (y2 < dst.get_y_size())
                    {
                        r += wc * scratch.r(x, y2);
                        g += wc * scratch.g(x, y2);
                        b += wc * scratch.b(x, y2);
                        n += wc;
                    }
                }
                dst.r(x, y) = r / n;
                dst.g(x, y) = g / n;
                dst.b(x, y) = b / n;
            }
        }

        return dst;
    }
#include "matrix.h"

#if !defined(FILTERS_HPP)
#define FILTERS_HPP

constexpr float maxX{1.33};
constexpr float pi{3.14159};
constexpr unsigned max_radius{1000};
void get_weights(int n, double *weights_out);

Matrix Blur(Matrix& m, int radius);

#endif
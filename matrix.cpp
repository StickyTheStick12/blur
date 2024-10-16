#include "matrix.h"

Matrix::Matrix(unsigned char* R, unsigned char* G, unsigned char* B, unsigned x_size, unsigned y_size, unsigned color_max)
    : R { R }
    , G { G }
    , B { B }
    , x_size { x_size }
    , y_size { y_size }
    , color_max { color_max }
{
}

Matrix::Matrix()
    : Matrix {
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        0,
    }
{
}

Matrix::Matrix(unsigned dimension)
    : R { new unsigned char[dimension * dimension] }
    , G { new unsigned char[dimension * dimension] }
    , B { new unsigned char[dimension * dimension] }
    , x_size { dimension }
    , y_size { dimension }
    , color_max { 0 }
{
}

Matrix::~Matrix()
{
    delete[] R;
    delete[] G;
    delete[] B;
}

unsigned Matrix::get_x_size() const
{
    return x_size;
}

unsigned Matrix::get_y_size() const
{
    return y_size;
}

unsigned Matrix::get_color_max() const
{
    return color_max;
}

unsigned char const* Matrix::get_R() const
{
    return R;
}

unsigned char const* Matrix::get_G() const
{
    return G;
}

unsigned char const* Matrix::get_B() const
{
    return B;
}

unsigned char Matrix::r(unsigned x, unsigned y) const
{
    return R[y * x_size + x];
}

unsigned char Matrix::g(unsigned x, unsigned y) const
{
    return G[y * x_size + x];
}

unsigned char Matrix::b(unsigned x, unsigned y) const
{
    return B[y * x_size + x];
}

unsigned char& Matrix::r(unsigned x, unsigned y)
{
    return R[y * x_size + x];
}

unsigned char& Matrix::g(unsigned x, unsigned y)
{
    return G[y * x_size + x];
}

unsigned char& Matrix::b(unsigned x, unsigned y)
{
    return B[y * x_size + x];
}
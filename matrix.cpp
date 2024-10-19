#include "matrix.h"

Matrix::Matrix(unsigned char* data, unsigned x_size, unsigned y_size, unsigned color_max)
    : data {data}
    , x_size { x_size }
    , y_size { y_size }
    , color_max { color_max }
{
}

Matrix::Matrix(unsigned xDim, unsigned yDim)
    : data { new unsigned char[xDim * yDim * 3]}
    , x_size { xDim }
    , y_size { yDim }
    , color_max { 0 }
{
}

Matrix::~Matrix()
{
    delete[] data;
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

unsigned char* Matrix::GetData() const {
    return data;
}

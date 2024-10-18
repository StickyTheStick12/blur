#if !defined(MATRIX_HPP)
#define MATRIX_HPP

class Matrix {
private:
    char* data;

    unsigned x_size;
    unsigned y_size;
    unsigned color_max;

public:
    Matrix(unsigned xDim, unsigned yDim);
    Matrix(char* data, unsigned x_size, unsigned y_size, unsigned color_max);
    ~Matrix();

    unsigned get_x_size() const;
    unsigned get_y_size() const;
    unsigned get_color_max() const;

    char* GetData() const;

};

#endif
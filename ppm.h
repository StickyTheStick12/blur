#include "matrix.h"
#include <sstream>

#if !defined(PPM_READER_HPP)
#define PPM_READER_HPP


void error(std::string op, std::string what);

constexpr unsigned max_dimension { 3000 };
constexpr unsigned max_pixels { max_dimension * max_dimension };
constexpr char const* magic_number { "P6" };

class Reader {
public:
    Matrix operator()(const std::string& filename);
};

class Writer {
public:
    void operator()(Matrix m, std::string filename);
};


#endif

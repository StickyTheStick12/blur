#include <string>

#include "matrix.h"

#if !defined(PPM_READER_HPP)
#define PPM_READER_HPP

Matrix Read(const int file, const long size);
void Write(const Matrix& m, const std::string& filename, long size);

#endif
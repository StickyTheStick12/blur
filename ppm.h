#include "matrix.h"

#if !defined(PPM_READER_HPP)
#define PPM_READER_HPP

Matrix Read(int file, off_t size);
void Write(const Matrix& m, const std::string& filename, off_t size);

#endif

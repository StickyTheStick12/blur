#include "matrix.h"
#include "ppm.h"
#include "filters.h"
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char const* argv[]) {
    const int file = open("im2.ppm", O_RDONLY);

    const off_t size = lseek(file, 0, SEEK_END);

    Matrix m = Read(file, size);

    const char* str = "12";
    int radius = 0;

    while (*str >= '0' && *str <= '9') {
        radius = radius * 10 + (*str - '0');
        str++;
    }

    Matrix blurred = Blur(m, radius);

    Write(blurred, "out.ppm", size);

    return 0;
}
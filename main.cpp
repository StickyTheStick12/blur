#include "matrix.h"
#include "ppm.h"
#include "filters.h"
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char const* argv[])
{
    const int file = open(argv[2], O_RDONLY);

    const off_t size = lseek(file, 0, SEEK_END);

    Matrix m = Read(file, size);

    const char* str = argv[1];
    int radius = 0;

    while (*str >= '0' && *str <= '9') {
        radius = radius * 10 + (*str - '0');
        str++;
    }

    auto blurred { Filter::blur(m, radius) };

    Write(blurred, argv[3], size);

    return 0;
}

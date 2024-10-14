#include "matrix.h"
#include "ppm.h"
#include "filters.h"
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char const* argv[]) {
    const int file = open(argv[2], O_RDONLY);

    const off_t size = lseek(file, 0, SEEK_END);

    Matrix m = Read(file, size);

    const char* str = argv[1];
    int radius = 0;

    while (*str >= '0' && *str <= '9') {
        radius = radius * 10 + (*str - '0');
        str++;
    }

    auto blurredOrig { blur(m, radius) };

    Matrix blurred = Blur(m, radius);

    auto R { blurredOrig.get_R() }, G { blurredOrig.get_G() }, B { blurredOrig.get_B() };
    auto it_Rorig { R }, it_Gorig { G }, it_Borig { B };

    auto Rf { blurred.get_R() }, Gf { blurred.get_G() }, Bf { blurred.get_B() };
    auto it_R { Rf }, it_G { Gf }, it_B { Bf };

    int index = 0;

    while (it_R < R + size && it_G < G + size && it_B < B + size) {
        if(*it_Rorig++ == *it_R++ || *it_Gorig++ == *it_G++ || *it_Borig++ == *it_B++) {
            {
                std::cout << "index: " << index << std::endl;
                std::cout << *it_R << std::endl;
                std::cout << *it_G << std::endl;
                std::cout << *it_B << std::endl;
                std::cout << *it_Rorig << std::endl;
                std::cout << *it_Gorig << std::endl;
                std::cout << *it_Borig << std::endl;
                std::cout << "ff" << std::endl;
                break;
            }
        }
        index++;
    }

    Write(blurred, argv[3], size);

    return 0;
}

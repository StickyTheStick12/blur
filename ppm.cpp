#include "ppm.h"
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>


Matrix Reader::operator()(const std::string& filename)
{
    int file = open(filename.c_str(), O_RDONLY);

    off_t size = lseek(file, 0, SEEK_END);

    char* mappedData = static_cast<char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0));

    madvise(mappedData, size, MADV_SEQUENTIAL);

    if(mappedData[0] != 'P' && mappedData[1] != '6') {
        std::abort();
    }

    mappedData += 3;

    if(mappedData[0] == '#') {
        while(mappedData[0] != '\n')
            mappedData++;

        mappedData++;
    }

    unsigned dimX = 0;
    unsigned dimY = 0;

    while(mappedData[0] != ' ') {
        dimX = dimX*10 + (mappedData[0] - '0');
        mappedData++;
    }

    mappedData++;

    while(mappedData[0] != '\n') {
        dimY = dimY*10 + (mappedData[0] - '0');
        mappedData++;
    }

    mappedData++;

    unsigned long totalSize = dimX*dimY;

    unsigned colorMax = 0;

    while(mappedData[0] != '\n') {
        colorMax = colorMax*10 + (mappedData[0] - '0');
        mappedData++;
    }

    unsigned char* R = new unsigned char[totalSize];
    unsigned char* G = new unsigned char[totalSize];
    unsigned char* B = new unsigned char[totalSize];

    for(unsigned i = 0; i < totalSize; ++i) {
        R[i] = mappedData[i*3];
        G[i] = mappedData[i*3+1];
        B[i] = mappedData[i*3+2];
    }

    munmap(mappedData, size);
    close(file);

    return { R, G, B, dimX, dimY, colorMax};
}

void error(std::string op, std::string what)
{
    std::cerr << "Encountered PPM error during " << op << ": " << what << std::endl;
}

void Writer::operator()(Matrix m, std::string filename)
{
    try {
        std::ofstream f {};

        f.open(filename);

        f << magic_number << std::endl;

        f << m.get_x_size() << " " << m.get_y_size() << std::endl;
        f << m.get_color_max() << std::endl;

        auto size { m.get_x_size() * m.get_y_size() };
        auto R { m.get_R() }, G { m.get_G() }, B { m.get_B() };
        auto it_R { R }, it_G { G }, it_B { B };

        while (it_R < R + size && it_G < G + size && it_B < B + size) {
            f << *it_R++
              << *it_G++
              << *it_B++;
        }

        f.close();
    } catch (std::runtime_error e) {
        error("writing", e.what());
    }
}

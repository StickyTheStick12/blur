#include "ppm.h"
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

Matrix Read(const int file, const long size)
{
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

    mappedData++;

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

    return Matrix{ R, G, B, dimX, dimY, colorMax};
}

void Write(const Matrix& m, const std::string& filename, const off_t fileSize)
{
    int file = open(filename.c_str(), O_RDWR | O_CREAT, 0666);

    ftruncate(file, fileSize);

    char* mappedOut = static_cast<char*>(mmap(nullptr, fileSize, PROT_WRITE, MAP_SHARED, file, 0));

    madvise(mappedOut, fileSize, MADV_SEQUENTIAL);

    mappedOut[0] = 'P';
    mappedOut[1] = '6';
    mappedOut[2] = '\n';

    mappedOut += 3;

    char buffer[4];

    int i = 0;
    int num = m.get_x_size();

    while(num > 0) {
        buffer[i++] = '0' + (num % 10);
        num /= 10;
    }

    int end = i-1;

    while(-1 < end) {
        *mappedOut++ = buffer[end--];
    }

    *mappedOut = ' ';
    mappedOut++;

    i = 0;
    num = m.get_x_size();

    while(num > 0) {
        buffer[i++] = '0' + (num % 10);
        num /= 10;
    }

    end = i-1;

    while(-1 < end) {
        *mappedOut++ = buffer[end--];
    }

    *mappedOut = '\n';
    mappedOut++;

    i = 0;
    num = m.get_color_max();

    while(num > 0) {
        buffer[i++] = '0' + (num % 10);
        num /= 10;
    }

    end = i-1;

    while(-1 < end) {
        *mappedOut++ = buffer[end--];
    }

    *mappedOut++ = '\n';

    unsigned size = m.get_x_size() * m.get_y_size();

    auto R { m.get_R() }, G { m.get_G() }, B { m.get_B() };
    auto it_R { R }, it_G { G }, it_B { B };

    while (it_R < R + size && it_G < G + size && it_B < B + size) {
        *mappedOut++ = *it_R++;
        *mappedOut++ = *it_G++;
        *mappedOut++ = *it_B++;
    }

    msync(mappedOut, fileSize, MS_SYNC);
    munmap(mappedOut, fileSize);
    close(file);
}

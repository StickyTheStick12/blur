#include "ppm.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

void fill(const std::string& filename)
{
    int file = open(filename.c_str(), O_RDONLY);

    off_t size = lseek(file, 0, SEEK_END);

    char* mappedData = static_cast<char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0));

    std::string magicNumber;

    int i = 0;

    while(mappedData[i] != '\n') {
        magicNumber += mappedData[i];
        i++;
    }

    if(mappedData[i] == '#')
        while(mappedData[i] != '\n')
            i++;

    std::string dimX;
    std::string dimY;

    while(mappedData[i] != ' ') {
        dimX += mappedData[i];
        i++;
    }

    while(mappedData[i] != '\n') {
        dimY += mappedData[i];
        i++;
    }

    std::string color;

    while(mappedData[i] != '\n') {
        color += mappedData[i];
        i++;
    }

    //todo: use a memory map as before, save to dynamic array



    //todo don't copy data from file? instead use the already defined char array from memory map. no need to copy if we use them as chars
}

std::string get_magic_number()
{
    //todo: move to void fill
    std::string magic {};

    std::getline(stream, magic);

    return magic;
}

std::pair<unsigned, unsigned> get_dimensions()
{
    //todo move to fill???
    std::string line {};

    while (std::getline(stream, line) && line[0] == '#')
        ;

    std::regex regex { "^(\\d+) (\\d+)$" };
    std::smatch matches {};

    std::regex_match(line, matches, regex);

    if (matches.ready()) {
        return { std::stoul(matches[1]), std::stoul(matches[2]) };
    } else {
        return { 0, 0 };
    }
}

unsigned get_color_max()
{
    //todo: move to fill
    std::string line {};

    std::getline(stream, line);

    std::regex regex { "^(\\d+)$" };
    std::smatch matches {};

    std::regex_match(line, matches, regex);

    if (matches.ready()) {
        return std::stoul(matches[1]);
    } else {
        return 0;
    }
}

std::tuple<unsigned char*, unsigned char*, unsigned char*> Reader::get_data(unsigned x_size, unsigned y_size)
{
    auto size { x_size * y_size };
    auto R { new char[size] }, G { new char[size] }, B { new char[size] };

    for (auto i { 0 }, read { 0 }; i < size; i++, read = 0) {
        stream.read(R + i, 1);
        read += stream.gcount();
        stream.read(G + i, 1);
        read += stream.gcount();
        stream.read(B + i, 1);
        read += stream.gcount();

        if (read != 3) {
            delete[] R;
            delete[] G;
            delete[] B;
            return { nullptr, nullptr, nullptr };
        }
    }

    return { reinterpret_cast<unsigned char*>(R), reinterpret_cast<unsigned char*>(G), reinterpret_cast<unsigned char*>(B) };
}

Matrix Reader::operator()(std::string filename)
{
    fill(filename);

    auto magic { get_magic_number() };

    auto [x_size, y_size] { get_dimensions() };

    auto total_size { x_size * y_size };

    auto color_max { get_color_max() };

    auto [R, G, B] { get_data(x_size, y_size) };

    stream.clear();
    return Matrix { R, G, B, x_size, y_size, color_max };
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

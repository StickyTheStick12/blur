#include "matrix.h"
#include "ppm.h"
#include "filters.h"
#include <cstdlib>

//TODO: change from stoul to the other library in pearson.

int main(int argc, char const* argv[])
{
    Reader reader {};
    Writer writer {};

    auto m { reader(argv[2]) };

    auto radius { static_cast<unsigned>(std::stoul(argv[1])) };

    auto blurred { Filter::blur(m, radius) };
    writer(blurred, argv[3]);

    return 0;
}

#include <memory>
#include <barrier>

#include "matrix.h"

#if !defined(FILTERS_HPP)
#define FILTERS_HPP

constexpr float maxX{1.33};
constexpr float pi{3.14159};
constexpr unsigned max_radius{1000};

void Blur(Matrix* m, std::shared_ptr<std::barrier<>> barrier, int radius, int startPos, int endPos);

#endif
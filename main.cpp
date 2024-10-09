#include <iostream>
#include <atomic>
#

int main() {
    unsigned total = 0;
    std::cout << "{";
    for (unsigned i = 0; i < 255; ++i) {
        std::cout << total << ", ";           // Starting index for i
        total += 255 - i;                   // The number of j iterations for this i
    }

    std::cout << "}" << std::endl;


    return 0;
}

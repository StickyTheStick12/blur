#include "matrix.h"
#include "ppm.h"
#include "filters.h"
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include <pthread.h>
#include <barrier>
#include <atomic>

//TODO. fix load instead of set?
//todo. compile with -funroll-loops
//todo. maybe use mmap array instead of copying it into a new array

std::atomic<int> threadIdCounter(0);

struct ThreadData {
    int nrThreads;
    int radius;
    Matrix* matrix;
    std::shared_ptr<std::barrier<>> barrier;
};

void* ThreadFunc(ThreadData* data)
{
    int threadId = threadIdCounter.fetch_add(1);

    int baseChunkSize = data->matrix->get_x_size()/data->nrThreads;
    int remainder = data->matrix->get_x_size()%data->nrThreads;

    int startPos = threadId*baseChunkSize + std::min(threadId, remainder);
    int chunkSize = baseChunkSize + (threadId < remainder ? 1 : 0);

    int endPos = startPos + chunkSize;

    Blur(data->matrix, data->barrier, data->radius, startPos, endPos);

    return nullptr;
}

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

    pthread_t threads[32];

    int nrThreads;
    if (argv[4][1] == '\0') {
        nrThreads = argv[4][0] - '0'; // Convert first character to integer
    } else {
        nrThreads = (argv[4][0] - '0') * 10 + (argv[4][1] - '0'); // Convert first two characters
    }

    ThreadData data;

    data.nrThreads = nrThreads;
    data.radius = radius;
    data.matrix = &m;
    data.barrier = std::make_shared<std::barrier<>>(nrThreads);

    for(int i = 0; i < nrThreads-1; ++i) {
        pthread_create(&threads[i], nullptr, (void * (*)(void *))ThreadFunc, &data);
    }

    ThreadFunc(&data);

    for(int i = 0; i < nrThreads-1; ++i)
        pthread_join(threads[i], nullptr);

    Write(m, argv[3], size);

    return 0;
}

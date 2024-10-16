#include "matrix.h"
#include "ppm.h"
#include "filters.h"
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include <pthread.h>


struct ThreadData {
    int nrThreads;
    int threadId;
    int radius;
    std::shared_ptr<Matrix> matrix;
};

void* ThreadFunc(ThreadData* data)
{
    int baseChunkSize = data->matrix->get_x_size()/data->nrThreads;
    int remainder = data->matrix->get_x_size()/data->nrThreads;

    int startPos = data->threadId*baseChunkSize + std::min(data->threadId, remainder);
    int chunkSize = baseChunkSize + (data->threadId < remainder ? 1 : 0);

    int endPos = startPos + chunkSize;

    Blur(data->matrix, data->radius, startPos, endPos);

    return nullptr;
}

int maini(int argc, char const* argv[]) {
    const int file = open("im2.ppm", O_RDONLY);

    const off_t size = lseek(file, 0, SEEK_END);

    Matrix m = Read(file, size);

    const char* str = "12";
    int radius = 0;

    while (*str >= '0' && *str <= '9') {
        radius = radius * 10 + (*str - '0');
        str++;
    }

    pthread_t threads[32];

    int nrThreads;
    if (argv[3][1] == '\0') {
        nrThreads = argv[3][0] - '0'; // Convert first character to integer
    } else {
        nrThreads = (argv[3][0] - '0') * 10 + (argv[3][1] - '0'); // Convert first two characters
    }


    ThreadData data;

    data.nrThreads = nrThreads;
    data.radius = radius;
    data.matrix = std::make_shared<Matrix>(m);

    for(int i = 0; i < nrThreads-1; ++i) {
        data.threadId=i;
        pthread_create(&threads[i], nullptr, (void * (*)(void *))ThreadFunc, &data);
    }

    data.threadId = nrThreads-1;

    ThreadFunc(&data);

    for(int i = 0; i < nrThreads-1; ++i)
        pthread_join(threads[i], nullptr);


    Write(m, "out.ppm", size);

    return 0;
}

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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long r = strtoull(argv[1], NULL, 10);
    unsigned long long k = strtoull(argv[2], NULL, 10);

    unsigned long long local_pixels = 0;
    unsigned long long batch_sum = 0;
    const unsigned long long BATCH_SIZE = 1048576;
    unsigned long long counter = 0;

    for (unsigned long long x = rank; x < r; x += size) {
        double diff = (double)(r - x) * (double)(r + x);
        unsigned long long y = (unsigned long long)sqrt(diff);

        if ((r - x) * (r + x) > y * y) {
            y++;
        }

        batch_sum += y;
        counter++;

        if (counter >= BATCH_SIZE) {
            local_pixels = (local_pixels + (batch_sum % k)) % k;
            batch_sum = 0;
            counter = 0;
        }
    }

    local_pixels = (local_pixels + (batch_sum % k)) % k;

    unsigned long long total_pixels = 0;
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        total_pixels %= k;
        printf("%llu\n", (4 * total_pixels) % k);
    }

    MPI_Finalize();
    return 0;
}
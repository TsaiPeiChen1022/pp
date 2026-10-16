#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long r = strtoull(argv[1], NULL, 10);
    unsigned long long k = strtoull(argv[2], NULL, 10);

    if (k == 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: k cannot be zero!\n");
        }
        MPI_Finalize();
        return 1;
    }

    unsigned long long local_pixels = 0;

    #pragma omp parallel reduction(+:local_pixels)
    {
        int omp_rank = omp_get_thread_num();
        int omp_size = omp_get_num_threads();

        unsigned long long total_workers = (unsigned long long)size * omp_size;
        unsigned long long global_id = (unsigned long long)rank * omp_size + omp_rank;

        unsigned long long thread_local_pixels = 0;
        unsigned long long batch_sum = 0;
        const unsigned long long BATCH_SIZE = 1048576;
        unsigned long long counter = 0;

        for (unsigned long long x = global_id; x < r; x += total_workers) {
            double diff = (double)(r - x) * (double)(r + x);
            unsigned long long y = (unsigned long long)sqrt(diff);

            if ((r - x) * (r + x) > y * y) {
                y++;
            }

            batch_sum += y;
            counter++;

            if (counter >= BATCH_SIZE) {
                thread_local_pixels = (thread_local_pixels + (batch_sum % k)) % k;
                batch_sum = 0;
                counter = 0;
            }
        }

        thread_local_pixels = (thread_local_pixels + (batch_sum % k)) % k;
        local_pixels = thread_local_pixels;
    }

    // 透過 MPI_Reduce 將所有行程的結果加總到 Rank 0
    unsigned long long total_pixels = 0;
    MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        total_pixels %= k;
        printf("%llu\n", (4 * total_pixels) % k);
    }

    MPI_Finalize();
    return 0;
}
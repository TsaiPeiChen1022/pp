#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = strtoull(argv[1], NULL, 10);
    unsigned long long k = strtoull(argv[2], NULL, 10);

    if (k == 0) {
        fprintf(stderr, "Error: k cannot be zero!\n");
        return 1;
    }

    int num_threads = 0;
    char *env_cpus = getenv("SLURM_CPUS_PER_TASK");
    if (env_cpus != NULL) {
        num_threads = atoi(env_cpus);
    }
    
    if (num_threads <= 0) {
        env_cpus = getenv("OMP_NUM_THREADS");
        if (env_cpus != NULL) {
            num_threads = atoi(env_cpus);
        }
    }

    if (num_threads <= 0) {
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    }

    if (num_threads <= 0) {
        num_threads = 4;
    }

    omp_set_num_threads(num_threads);

    unsigned long long total_pixels = 0;

    #pragma omp parallel reduction(+:total_pixels)
    {
        int rank = omp_get_thread_num();
        int size = omp_get_num_threads();

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
        
        total_pixels = local_pixels;
    }

    printf("%llu\n", (4 * total_pixels) % k);

    return 0;
}
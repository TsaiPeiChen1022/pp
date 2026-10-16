#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

// 定義傳遞給執行緒的參數結構
typedef struct {
    int thread_id;
    int num_threads;
    unsigned long long r;
    unsigned long long k;
    unsigned long long result_pixels;
} thread_data_t;

// 每個執行緒執行的計算函數
void *calculate(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    unsigned long long r = data->r;
    unsigned long long k = data->k;
    int rank = data->thread_id;
    int size = data->num_threads;

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
    data->result_pixels = local_pixels;

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = strtoull(argv[1], NULL, 10);
    unsigned long long k = strtoull(argv[2], NULL, 10);

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

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = (thread_data_t *)malloc(num_threads * sizeof(thread_data_t));

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].r = r;
        thread_data[i].k = k;
        thread_data[i].result_pixels = 0;

        pthread_create(&threads[i], NULL, calculate, (void *)&thread_data[i]);
    }

    unsigned long long total_pixels = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_pixels = (total_pixels + thread_data[i].result_pixels) % k;
    }

    printf("%llu\n", (4 * total_pixels) % k);

    free(threads);
    free(thread_data);

    return 0;
}
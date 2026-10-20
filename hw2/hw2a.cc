#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

typedef struct {
    int thread_id;
    int num_threads;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
    int* image;
} thread_data_t;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* mandelbrot_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    for (int j = data->thread_id; j < data->height; j += data->num_threads) {
        double y0_val = j * ((data->upper - data->lower) / data->height) + data->lower;
        double dx = (data->right - data->left) / data->width;
        double left = data->left;

        for (int i = 0; i < data->width; ++i) {
            double x0 = i * dx + left;
            int repeats = 0;
            double x = 0.0, y = 0.0, length_squared = 0.0;
            while (repeats < data->iters && length_squared < 4.0) {
                double temp = x * x - y * y + x0;
                y = 2.0 * x * y + y0_val;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            data->image[j * data->width + i] = repeats;
        }
    }
    
    return NULL;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int num_threads = 0;
    char *env_cpus = getenv("SLURM_CPUS_PER_TASK");
    if (env_cpus != NULL) {
        num_threads = atoi(env_cpus);
    }
    if (num_threads <= 0) {
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    }
    assert(num_threads > 0);

    struct timespec start_calculation_time, end_calculation_time;
    struct timespec start_write_time, end_write_time;

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = (thread_data_t *)malloc(num_threads * sizeof(thread_data_t));

    clock_gettime(CLOCK_MONOTONIC, &start_calculation_time);

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].iters = iters;
        thread_data[i].left = left;
        thread_data[i].right = right;
        thread_data[i].lower = lower;
        thread_data[i].upper = upper;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].image = image;

        pthread_create(&threads[i], NULL, mandelbrot_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_calculation_time);

    free(threads);
    free(thread_data);

    /* draw and cleanup */
    clock_gettime(CLOCK_MONOTONIC, &start_write_time);
    write_png(filename, iters, width, height, image);
    free(image);
    clock_gettime(CLOCK_MONOTONIC, &end_write_time);

    double calc_time = (end_calculation_time.tv_sec - start_calculation_time.tv_sec) + 
                       (end_calculation_time.tv_nsec - start_calculation_time.tv_nsec) / 1e9;
    double write_time = (end_write_time.tv_sec - start_write_time.tv_sec) + 
                        (end_write_time.tv_nsec - start_write_time.tv_nsec) / 1e9;
    printf("Calculation time: %.9f seconds\n", calc_time);
    printf("Write time: %.9f seconds\n", write_time);
    return 0;
}
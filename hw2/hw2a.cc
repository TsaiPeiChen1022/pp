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
#include <immintrin.h>

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
    int* global_row;
    pthread_mutex_t* row_mutex;
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

    const __m512d v_four = _mm512_set1_pd(4.0);

    while (1) {
        /* 1. 動態取得下一行要計算的工作 */
        pthread_mutex_lock(data->row_mutex);
        int j = *(data->global_row);
        (*(data->global_row))++;
        pthread_mutex_unlock(data->row_mutex);

        if (j >= data->height) {
            break;
        }

        double y0_val = j * ((data->upper - data->lower) / data->height) + data->lower;
        __m512d y0 = _mm512_set1_pd(y0_val); 

        double dx = (data->right - data->left) / data->width;
        double left = data->left;

        int i = 0;
        /* 2. AVX-512 3組交錯迴圈：一次處理 24 個像素 (3個 zmm 寬度) */
        for (; i <= data->width - 24; i += 24) {
            double x0_vals_A[8], x0_vals_B[8], x0_vals_C[8];
            for (int k = 0; k < 8; ++k) {
                x0_vals_A[k] = (i + k) * dx + left;
                x0_vals_B[k] = (i + 8 + k) * dx + left;
                x0_vals_C[k] = (i + 16 + k) * dx + left;
            }
            __m512d x0_A = _mm512_loadu_pd(x0_vals_A);
            __m512d x0_B = _mm512_loadu_pd(x0_vals_B);
            __m512d x0_C = _mm512_loadu_pd(x0_vals_C);

            __m512d x_A = _mm512_setzero_pd(), y_A = _mm512_setzero_pd(), lsq_A = _mm512_setzero_pd();
            __m512d x_B = _mm512_setzero_pd(), y_B = _mm512_setzero_pd(), lsq_B = _mm512_setzero_pd();
            __m512d x_C = _mm512_setzero_pd(), y_C = _mm512_setzero_pd(), lsq_C = _mm512_setzero_pd();
            
            int repeats_A[8] = {0};
            int repeats_B[8] = {0};
            int repeats_C[8] = {0};
            int iter = 0;

            while (iter < data->iters) {
                __mmask8 mask_A = _mm512_cmp_pd_mask(lsq_A, v_four, _MM_CMPINT_LT);
                __mmask8 mask_B = _mm512_cmp_pd_mask(lsq_B, v_four, _MM_CMPINT_LT);
                __mmask8 mask_C = _mm512_cmp_pd_mask(lsq_C, v_four, _MM_CMPINT_LT);
                
                /* 如果三組都已經全部發散，直接跳出迴圈 */
                if (mask_A == 0 && mask_B == 0 && mask_C == 0) break;

                /* A 組狀態更新 */
                if (mask_A != 0) {
                    for (int k = 0; k < 8; ++k) {
                        if ((mask_A >> k) & 1) repeats_A[k]++;
                    }
                    __m512d x2_A = _mm512_mul_pd(x_A, x_A);
                    __m512d y2_A = _mm512_mul_pd(y_A, y_A);
                    __m512d xy_A = _mm512_mul_pd(x_A, y_A);
                    __m512d temp_A = _mm512_add_pd(_mm512_sub_pd(x2_A, y2_A), x0_A);
                    __m512d next_y_A = _mm512_add_pd(_mm512_add_pd(xy_A, xy_A), y0);
                    x_A = _mm512_mask_mov_pd(x_A, mask_A, temp_A);
                    y_A = _mm512_mask_mov_pd(y_A, mask_A, next_y_A);
                    lsq_A = _mm512_add_pd(_mm512_mul_pd(x_A, x_A), _mm512_mul_pd(y_A, y_A));
                }

                /* B 組狀態更新 */
                if (mask_B != 0) {
                    for (int k = 0; k < 8; ++k) {
                        if ((mask_B >> k) & 1) repeats_B[k]++;
                    }
                    __m512d x2_B = _mm512_mul_pd(x_B, x_B);
                    __m512d y2_B = _mm512_mul_pd(y_B, y_B);
                    __m512d xy_B = _mm512_mul_pd(x_B, y_B);
                    __m512d temp_B = _mm512_add_pd(_mm512_sub_pd(x2_B, y2_B), x0_B);
                    __m512d next_y_B = _mm512_add_pd(_mm512_add_pd(xy_B, xy_B), y0);
                    x_B = _mm512_mask_mov_pd(x_B, mask_B, temp_B);
                    y_B = _mm512_mask_mov_pd(y_B, mask_B, next_y_B);
                    lsq_B = _mm512_add_pd(_mm512_mul_pd(x_B, x_B), _mm512_mul_pd(y_B, y_B));
                }

                /* C 組狀態更新 */
                if (mask_C != 0) {
                    for (int k = 0; k < 8; ++k) {
                        if ((mask_C >> k) & 1) repeats_C[k]++;
                    }
                    __m512d x2_C = _mm512_mul_pd(x_C, x_C);
                    __m512d y2_C = _mm512_mul_pd(y_C, y_C);
                    __m512d xy_C = _mm512_mul_pd(x_C, y_C);
                    __m512d temp_C = _mm512_add_pd(_mm512_sub_pd(x2_C, y2_C), x0_C);
                    __m512d next_y_C = _mm512_add_pd(_mm512_add_pd(xy_C, xy_C), y0);
                    x_C = _mm512_mask_mov_pd(x_C, mask_C, temp_C);
                    y_C = _mm512_mask_mov_pd(y_C, mask_C, next_y_C);
                    lsq_C = _mm512_add_pd(_mm512_mul_pd(x_C, x_C), _mm512_mul_pd(y_C, y_C));
                }

                iter++;
            }

            /* 寫回 A、B、C 三組結果 */
            for (int k = 0; k < 8; ++k) {
                data->image[j * data->width + (i + k)] = repeats_A[k];
                data->image[j * data->width + (i + 8 + k)] = repeats_B[k];
                data->image[j * data->width + (i + 16 + k)] = repeats_C[k];
            }
        }

        /* 3. 中間尾數處理 (處理剩下的 >= 8 且 < 24 個像素) */
        for (; i <= data->width - 8; i += 8) {
            double x0_vals[8];
            for (int k = 0; k < 8; ++k) {
                x0_vals[k] = (i + k) * dx + left;
            }
            __m512d x0 = _mm512_loadu_pd(x0_vals);
            __m512d x = _mm512_setzero_pd();
            __m512d y = _mm512_setzero_pd();
            __m512d length_squared = _mm512_setzero_pd();
            int repeats[8] = {0}; 
            int iter = 0;

            while (iter < data->iters) {
                __mmask8 active_mask = _mm512_cmp_pd_mask(length_squared, v_four, _MM_CMPINT_LT);
                if (active_mask == 0) break;
                for (int k = 0; k < 8; ++k) {
                    if ((active_mask >> k) & 1) repeats[k]++;
                }
                __m512d x2 = _mm512_mul_pd(x, x);
                __m512d y2 = _mm512_mul_pd(y, y);
                __m512d xy = _mm512_mul_pd(x, y);
                __m512d temp = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0);
                __m512d next_y = _mm512_add_pd(_mm512_add_pd(xy, xy), y0);
                x = _mm512_mask_mov_pd(x, active_mask, temp);
                y = _mm512_mask_mov_pd(y, active_mask, next_y);
                length_squared = _mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y));
                iter++;
            }
            for (int k = 0; k < 8; ++k) {
                data->image[j * data->width + (i + k)] = repeats[k];
            }
        }

        /* 4. 純量尾數補齊 (Tail Handling) */
        for (; i < data->width; ++i) {
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

    /* 初始化動態排程的全域變數與 Mutex */
    int global_row = 0;
    pthread_mutex_t row_mutex;
    pthread_mutex_init(&row_mutex, NULL);

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
        thread_data[i].global_row = &global_row;
        thread_data[i].row_mutex = &row_mutex;

        pthread_create(&threads[i], NULL, mandelbrot_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_calculation_time);

    /* 銷毀 Mutex 與釋放動態記憶體 */
    pthread_mutex_destroy(&row_mutex);
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
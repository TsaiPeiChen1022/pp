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
#include <unistd.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

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
    png_bytep image_data = (png_bytep)calloc(row_size, (size_t)height);
    png_bytep* rows = (png_bytep*)malloc(height * sizeof(png_bytep));
    for (int y = 0; y < height; ++y) rows[y] = image_data + (size_t)y * row_size;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = rows[y] + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
    }

    png_write_image(png_ptr, rows);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    free(rows);
    free(image_data);
    fclose(fp);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

    struct timespec start_calculation_time, end_calculation_time;
    struct timespec start_write_time, end_write_time;

    /* 計算每個 MPI Process 負責的行數與範圍（處理除不盡餘數的問題） */
    int rows_per_proc = height / world_size;
    int remainder = height % world_size;
    int my_rows = rows_per_proc + (world_rank < remainder ? 1 : 0);
    
    // 計算每個 rank 的起始行與接收 counts / displs（用於 MPI_Gatherv）
    int* recvcounts = NULL;
    int* displs = NULL;
    if (world_rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        int accumulated_displs = 0;
        for (int i = 0; i < world_size; ++i) {
            int r = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = r * width;
            displs[i] = accumulated_displs;
            accumulated_displs += recvcounts[i];
        }
    }

    int* local_image = (int*)malloc(my_rows * width * sizeof(int));
    assert(local_image);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &start_calculation_time);

    const double y_diff = (upper - lower) / height;
    const double x_diff = (right - left) / width;
    const __m512d v_four = _mm512_set1_pd(4.0);

    /* OpenMP 動態排程平行計算本地負責的行 */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int r = 0; r < my_rows; ++r) {
        //還原全域的真實行號 j
        int j;
        if (world_rank < remainder) {
            j = world_rank * (rows_per_proc + 1) + r;
        } else {
            j = remainder * (rows_per_proc + 1) + (world_rank - remainder) * rows_per_proc + r;
        }

        double y0_val = j * y_diff + lower;
        __m512d y0 = _mm512_set1_pd(y0_val);

        int i = 0;
        /* AVX-512 4組交錯核心 (32 pixels per iteration) */
        for (; i <= width - 32; i += 32) {
            double x0_vals_A[8], x0_vals_B[8], x0_vals_C[8], x0_vals_D[8];
            for (int k = 0; k < 8; ++k) {
                x0_vals_A[k] = (i + k) * x_diff + left;
                x0_vals_B[k] = (i + 8 + k) * x_diff + left;
                x0_vals_C[k] = (i + 16 + k) * x_diff + left;
                x0_vals_D[k] = (i + 24 + k) * x_diff + left;
            }
            __m512d x0_A = _mm512_loadu_pd(x0_vals_A);
            __m512d x0_B = _mm512_loadu_pd(x0_vals_B);
            __m512d x0_C = _mm512_loadu_pd(x0_vals_C);
            __m512d x0_D = _mm512_loadu_pd(x0_vals_D);

            __m512d x_A = _mm512_setzero_pd(), y_A = _mm512_setzero_pd(), lsq_A = _mm512_setzero_pd();
            __m512d x_B = _mm512_setzero_pd(), y_B = _mm512_setzero_pd(), lsq_B = _mm512_setzero_pd();
            __m512d x_C = _mm512_setzero_pd(), y_C = _mm512_setzero_pd(), lsq_C = _mm512_setzero_pd();
            __m512d x_D = _mm512_setzero_pd(), y_D = _mm512_setzero_pd(), lsq_D = _mm512_setzero_pd();
            
            int repeats_A[8] = {0}, repeats_B[8] = {0}, repeats_C[8] = {0}, repeats_D[8] = {0};
            int iter = 0;

            while (iter < iters) {
                __mmask8 mask_A = _mm512_cmp_pd_mask(lsq_A, v_four, _MM_CMPINT_LT);
                __mmask8 mask_B = _mm512_cmp_pd_mask(lsq_B, v_four, _MM_CMPINT_LT);
                __mmask8 mask_C = _mm512_cmp_pd_mask(lsq_C, v_four, _MM_CMPINT_LT);
                __mmask8 mask_D = _mm512_cmp_pd_mask(lsq_D, v_four, _MM_CMPINT_LT);
                
                if (mask_A == 0 && mask_B == 0 && mask_C == 0 && mask_D == 0) break;

                if (mask_A != 0) {
                    for (int k = 0; k < 8; ++k) if ((mask_A >> k) & 1) repeats_A[k]++;
                    __m512d x2_A = _mm512_mul_pd(x_A, x_A);
                    __m512d y2_A = _mm512_mul_pd(y_A, y_A);
                    __m512d xy_A = _mm512_mul_pd(x_A, y_A);
                    __m512d temp_A = _mm512_add_pd(_mm512_sub_pd(x2_A, y2_A), x0_A);
                    __m512d next_y_A = _mm512_add_pd(_mm512_add_pd(xy_A, xy_A), y0);
                    x_A = _mm512_mask_mov_pd(x_A, mask_A, temp_A);
                    y_A = _mm512_mask_mov_pd(y_A, mask_A, next_y_A);
                    lsq_A = _mm512_add_pd(_mm512_mul_pd(x_A, x_A), _mm512_mul_pd(y_A, y_A));
                }

                if (mask_B != 0) {
                    for (int k = 0; k < 8; ++k) if ((mask_B >> k) & 1) repeats_B[k]++;
                    __m512d x2_B = _mm512_mul_pd(x_B, x_B);
                    __m512d y2_B = _mm512_mul_pd(y_B, y_B);
                    __m512d xy_B = _mm512_mul_pd(x_B, y_B);
                    __m512d temp_B = _mm512_add_pd(_mm512_sub_pd(x2_B, y2_B), x0_B);
                    __m512d next_y_B = _mm512_add_pd(_mm512_add_pd(xy_B, xy_B), y0);
                    x_B = _mm512_mask_mov_pd(x_B, mask_B, temp_B);
                    y_B = _mm512_mask_mov_pd(y_B, mask_B, next_y_B);
                    lsq_B = _mm512_add_pd(_mm512_mul_pd(x_B, x_B), _mm512_mul_pd(y_B, y_B));
                }

                if (mask_C != 0) {
                    for (int k = 0; k < 8; ++k) if ((mask_C >> k) & 1) repeats_C[k]++;
                    __m512d x2_C = _mm512_mul_pd(x_C, x_C);
                    __m512d y2_C = _mm512_mul_pd(y_C, y_C);
                    __m512d xy_C = _mm512_mul_pd(x_C, y_C);
                    __m512d temp_C = _mm512_add_pd(_mm512_sub_pd(x2_C, y2_C), x0_C);
                    __m512d next_y_C = _mm512_add_pd(_mm512_add_pd(xy_C, xy_C), y0);
                    x_C = _mm512_mask_mov_pd(x_C, mask_C, temp_C);
                    y_C = _mm512_mask_mov_pd(y_C, mask_C, next_y_C);
                    lsq_C = _mm512_add_pd(_mm512_mul_pd(x_C, x_C), _mm512_mul_pd(y_C, y_C));
                }

                if (mask_D != 0) {
                    for (int k = 0; k < 8; ++k) if ((mask_D >> k) & 1) repeats_D[k]++;
                    __m512d x2_D = _mm512_mul_pd(x_D, x_D);
                    __m512d y2_D = _mm512_mul_pd(y_D, y_D);
                    __m512d xy_D = _mm512_mul_pd(x_D, y_D);
                    __m512d temp_D = _mm512_add_pd(_mm512_sub_pd(x2_D, y2_D), x0_D);
                    __m512d next_y_D = _mm512_add_pd(_mm512_add_pd(xy_D, xy_D), y0);
                    x_D = _mm512_mask_mov_pd(x_D, mask_D, temp_D);
                    y_D = _mm512_mask_mov_pd(y_D, mask_D, next_y_D);
                    lsq_D = _mm512_add_pd(_mm512_mul_pd(x_D, x_D), _mm512_mul_pd(y_D, y_D));
                }

                iter++;
            }

            int row_offset = r * width;
            for (int k = 0; k < 8; ++k) {
                local_image[row_offset + (i + k)] = repeats_A[k];
                local_image[row_offset + (i + 8 + k)] = repeats_B[k];
                local_image[row_offset + (i + 16 + k)] = repeats_C[k];
                local_image[row_offset + (i + 24 + k)] = repeats_D[k];
            }
        }

        /* 尾數處理 */
        for (; i <= width - 8; i += 8) {
            double x0_vals[8];
            for (int k = 0; k < 8; ++k) x0_vals[k] = (i + k) * x_diff + left;
            __m512d x0 = _mm512_loadu_pd(x0_vals);
            __m512d x = _mm512_setzero_pd(), y = _mm512_setzero_pd(), length_squared = _mm512_setzero_pd();
            int repeats[8] = {0}, iter = 0;

            while (iter < iters) {
                __mmask8 active_mask = _mm512_cmp_pd_mask(length_squared, v_four, _MM_CMPINT_LT);
                if (active_mask == 0) break;
                for (int k = 0; k < 8; ++k) if ((active_mask >> k) & 1) repeats[k]++;
                __m512d x2 = _mm512_mul_pd(x, x), y2 = _mm512_mul_pd(y, y), xy = _mm512_mul_pd(x, y);
                __m512d temp = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0);
                __m512d next_y = _mm512_add_pd(_mm512_add_pd(xy, xy), y0);
                x = _mm512_mask_mov_pd(x, active_mask, temp);
                y = _mm512_mask_mov_pd(y, active_mask, next_y);
                length_squared = _mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y));
                iter++;
            }
            int row_offset = r * width;
            for (int k = 0; k < 8; ++k) local_image[row_offset + (i + k)] = repeats[k];
        }

        for (; i < width; ++i) {
            double x0 = i * x_diff + left;
            int repeats = 0;
            double x = 0.0, y = 0.0, length_squared = 0.0;
            while (repeats < iters && length_squared < 4.0) {
                double temp = x * x - y * y + x0;
                y = 2.0 * x * y + y0_val;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            local_image[r * width + i] = repeats;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end_calculation_time);

    /* 收集所有行程的計算結果到 Rank 0 */
    int* image = NULL;
    if (world_rank == 0) {
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);
    }

    MPI_Gatherv(local_image, my_rows * width, MPI_INT,
                image, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* 輸出 PNG 檔案 */
    if (world_rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start_write_time);
        write_png(filename, iters, width, height, image);
        clock_gettime(CLOCK_MONOTONIC, &end_write_time);
        free(image);

        double calc_time = (end_calculation_time.tv_sec - start_calculation_time.tv_sec) + 
                           (end_calculation_time.tv_nsec - start_calculation_time.tv_nsec) / 1e9;
        double write_time = (end_write_time.tv_sec - start_write_time.tv_sec) + 
                            (end_write_time.tv_nsec - start_write_time.tv_nsec) / 1e9;
        printf("Calculation time: %.9f seconds\n", calc_time);
        printf("Write time: %.9f seconds\n", write_time);
    }

    free(local_image);
    if (world_rank == 0) {
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
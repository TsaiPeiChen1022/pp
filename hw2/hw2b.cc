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
#include <pthread.h>
#include <atomic>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

#define TH_FIELDS 6   
#define RK_FIELDS 7   

#define TAG_REQ  1
#define TAG_RESP 2
#define TAG_DATA 3


static std::atomic<int> g_next_row(0);
static int g_height = 0;
static int g_world_size = 1;

static inline int grab_chunk(int chunk, int* start) {
    int s = g_next_row.fetch_add(chunk, std::memory_order_relaxed);
    if (s >= g_height) return 0;
    *start = s;
    return std::min(chunk, g_height - s);
}

static void* server_main(void*) {
    int finished = 0;
    const int need = g_world_size - 1;
    struct timespec nap = {0, 20000};   
    while (finished < need) {
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQ, MPI_COMM_WORLD, &flag, &st);
        if (!flag) { nanosleep(&nap, NULL); continue; }

        int req_chunk;
        MPI_Recv(&req_chunk, 1, MPI_INT, st.MPI_SOURCE, TAG_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int resp[2] = {0, 0};
        resp[1] = grab_chunk(req_chunk, &resp[0]);
        if (resp[1] == 0) { resp[0] = 0; finished++; }
        MPI_Send(resp, 2, MPI_INT, st.MPI_SOURCE, TAG_RESP, MPI_COMM_WORLD);
    }
    return NULL;
}

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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size > 1 && provided < MPI_THREAD_SERIALIZED) {
        if (world_rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_SERIALIZED (provided=%d)\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    g_height = height;
    g_world_size = world_size;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_total_start = MPI_Wtime();

    const double y_diff = (upper - lower) / height;
    const double x_diff = (right - left) / width;
    const __m512d v_four = _mm512_set1_pd(4.0);

    const int max_threads = omp_get_max_threads();
    int chunk_size = 2 * max_threads;
    if (const char* env = getenv("HW2_CHUNK")) {
        int v = atoi(env);
        if (v > 0) chunk_size = v;
    }
    if (chunk_size < 1) chunk_size = 1;


    int* local_image = (int*)malloc((size_t)width * height * sizeof(int));
    assert(local_image);
    int* image = (world_rank == 0) ? local_image : NULL;

    double*    t_compute = (double*)calloc(max_threads, sizeof(double));
    double*    t_idle    = (double*)calloc(max_threads, sizeof(double));
    double*    t_sched   = (double*)calloc(max_threads, sizeof(double));
    long long* t_rows    = (long long*)calloc(max_threads, sizeof(long long));
    int*       t_cpu     = (int*)malloc(max_threads * sizeof(int));
    long long* t_migr    = (long long*)calloc(max_threads, sizeof(long long));
    for (int i = 0; i < max_threads; ++i) t_cpu[i] = -1;
    int actual_threads = 1;

    double region_wall = 0.0;

    int local_next = 0, local_end = 0, rank_done = 0;
    long long chunks_fetched = 0;
    std::vector<int> chunk_list;   

    MPI_Barrier(MPI_COMM_WORLD);

    pthread_t server_tid;
    bool server_started = false;
    if (world_rank == 0 && world_size > 1) {
        pthread_create(&server_tid, NULL, server_main, NULL);
        server_started = true;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        #pragma omp single
        actual_threads = omp_get_num_threads();

        const double thread_region_start = omp_get_wtime();
        double active_time = 0.0;
        double sched_time = 0.0;
        long long rows_handled = 0;
        long long migrations = 0;
        int prev_cpu = sched_getcpu();

        while (true) {
            const double sched_start = omp_get_wtime();
            int j = -1;
            #pragma omp critical(row_sched)
            {
                if (local_next >= local_end && !rank_done) {
                    int s = 0, c = 0;
                    if (world_rank == 0) {
                        c = grab_chunk(chunk_size, &s);
                    } else {
                        int req = chunk_size;
                        int resp[2];
                        MPI_Send(&req, 1, MPI_INT, 0, TAG_REQ, MPI_COMM_WORLD);
                        MPI_Recv(resp, 2, MPI_INT, 0, TAG_RESP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        s = resp[0];
                        c = resp[1];
                    }
                    if (c == 0) {
                        rank_done = 1;
                    } else {
                        local_next = s;
                        local_end = s + c;
                        chunks_fetched++;
                        if (world_rank != 0) { chunk_list.push_back(s); chunk_list.push_back(c); }
                    }
                }
                if (local_next < local_end) j = local_next++;
            }
            sched_time += omp_get_wtime() - sched_start;
            if (j < 0) break;

            const double iter_start = omp_get_wtime();
            int cur_cpu = sched_getcpu();
            if (cur_cpu != prev_cpu) { migrations++; prev_cpu = cur_cpu; }
            rows_handled++;

            double y0_val = j * y_diff + lower;
            __m512d y0 = _mm512_set1_pd(y0_val);
            const int row_offset = j * width;

            int i = 0;
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
                    __mmask8 mask_A = _mm512_cmp_pd_mask(lsq_A, v_four, _CMP_LT_OS);
                    __mmask8 mask_B = _mm512_cmp_pd_mask(lsq_B, v_four, _CMP_LT_OS);
                    __mmask8 mask_C = _mm512_cmp_pd_mask(lsq_C, v_four, _CMP_LT_OS);
                    __mmask8 mask_D = _mm512_cmp_pd_mask(lsq_D, v_four, _CMP_LT_OS);

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

                for (int k = 0; k < 8; ++k) {
                    local_image[row_offset + (i + k)] = repeats_A[k];
                    local_image[row_offset + (i + 8 + k)] = repeats_B[k];
                    local_image[row_offset + (i + 16 + k)] = repeats_C[k];
                    local_image[row_offset + (i + 24 + k)] = repeats_D[k];
                }
            }

            for (; i <= width - 8; i += 8) {
                double x0_vals[8];
                for (int k = 0; k < 8; ++k) x0_vals[k] = (i + k) * x_diff + left;
                __m512d x0 = _mm512_loadu_pd(x0_vals);
                __m512d x = _mm512_setzero_pd(), y = _mm512_setzero_pd(), length_squared = _mm512_setzero_pd();
                int repeats[8] = {0}, iter = 0;

                while (iter < iters) {
                    __mmask8 active_mask = _mm512_cmp_pd_mask(length_squared, v_four, _CMP_LT_OS);
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
                local_image[row_offset + i] = repeats;
            }

            active_time += (omp_get_wtime() - iter_start);
        }

        t_compute[tid] = active_time;
        t_sched[tid]   = sched_time;
        t_rows[tid]    = rows_handled;
        t_cpu[tid]     = sched_getcpu();
        t_migr[tid]    = migrations;

        #pragma omp barrier
        const double thread_region_end = omp_get_wtime();

        t_idle[tid] = (thread_region_end - thread_region_start) - active_time;

        #pragma omp master
        region_wall = thread_region_end - thread_region_start;
    }

    const double t_bar_start = MPI_Wtime();
    if (server_started) pthread_join(server_tid, NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    const double barrier_wait = MPI_Wtime() - t_bar_start;

    const double t_gather_start = MPI_Wtime();

    int my_nchunks = (world_rank == 0) ? 0 : (int)(chunk_list.size() / 2);
    int* all_n = NULL;
    int* lcounts = NULL;
    int* ldispls = NULL;
    int* all_lists = NULL;
    int total_chunks = 0;
    if (world_rank == 0) {
        all_n = (int*)malloc(world_size * sizeof(int));
        lcounts = (int*)malloc(world_size * sizeof(int));
        ldispls = (int*)malloc(world_size * sizeof(int));
    }
    MPI_Gather(&my_nchunks, 1, MPI_INT, all_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            lcounts[i] = all_n[i] * 2;
            ldispls[i] = total_chunks * 2;
            total_chunks += all_n[i];
        }
        all_lists = (int*)malloc((size_t)(total_chunks > 0 ? total_chunks : 1) * 2 * sizeof(int));
    }
    MPI_Gatherv(chunk_list.data(), my_nchunks * 2, MPI_INT,
                all_lists, lcounts, ldispls, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<MPI_Request> reqs;
    if (world_rank == 0) {
        reqs.reserve(total_chunks);
        for (int src = 1; src < world_size; ++src) {
            for (int i = 0; i < all_n[src]; ++i) {
                int s = all_lists[ldispls[src] + 2 * i];
                int c = all_lists[ldispls[src] + 2 * i + 1];
                MPI_Request rq;
                MPI_Irecv(image + (size_t)s * width, c * width, MPI_INT, src, TAG_DATA, MPI_COMM_WORLD, &rq);
                reqs.push_back(rq);
            }
        }
    } else {
        reqs.reserve(my_nchunks);
        for (int i = 0; i < my_nchunks; ++i) {
            int s = chunk_list[2 * i];
            int c = chunk_list[2 * i + 1];
            MPI_Request rq;
            MPI_Isend(local_image + (size_t)s * width, c * width, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, &rq);
            reqs.push_back(rq);
        }
    }
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    const double gather_time = MPI_Wtime() - t_gather_start;

    if (world_rank == 0) {
        free(all_n); free(lcounts); free(ldispls); free(all_lists);
    } else {
        free(local_image);
        local_image = NULL;
    }

    double io_time = 0.0;
    if (world_rank == 0) {
        const double t_io_start = MPI_Wtime();
        write_png(filename, iters, width, height, image);
        io_time = MPI_Wtime() - t_io_start;
        free(image);
    }

    const double t_fw_start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    const double final_wait = MPI_Wtime() - t_fw_start;

    const double total_time = MPI_Wtime() - t_total_start;

    double* th_local = (double*)calloc((size_t)max_threads * TH_FIELDS, sizeof(double));
    for (int t = 0; t < max_threads; ++t) {
        th_local[t * TH_FIELDS + 0] = (double)t_cpu[t];
        th_local[t * TH_FIELDS + 1] = (double)t_rows[t];
        th_local[t * TH_FIELDS + 2] = t_compute[t];
        th_local[t * TH_FIELDS + 3] = t_idle[t];
        th_local[t * TH_FIELDS + 4] = (double)t_migr[t];
        th_local[t * TH_FIELDS + 5] = t_sched[t];
    }
    double rk_local[RK_FIELDS] = { region_wall, barrier_wait, gather_time, io_time,
                                   final_wait, total_time, (double)chunks_fetched };

    double* th_all = NULL;
    double* rk_all = NULL;
    if (world_rank == 0) {
        th_all = (double*)malloc((size_t)world_size * max_threads * TH_FIELDS * sizeof(double));
        rk_all = (double*)malloc((size_t)world_size * RK_FIELDS * sizeof(double));
    }
    MPI_Gather(th_local, max_threads * TH_FIELDS, MPI_DOUBLE,
               th_all,   max_threads * TH_FIELDS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(rk_local, RK_FIELDS, MPI_DOUBLE,
               rk_all,   RK_FIELDS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("CSV_HEADER,procs,threads,rank,tid,cpu,rows,compute_ms,idle_ms,migrations,"
               "region_ms,barrier_wait_ms,gatherv_ms,io_ms,final_wait_ms,total_ms,sched_ms,chunks\n");
        for (int rk = 0; rk < world_size; ++rk) {
            const double* R = rk_all + rk * RK_FIELDS;
            for (int t = 0; t < actual_threads; ++t) {
                const double* T = th_all + ((size_t)rk * max_threads + t) * TH_FIELDS;
                printf("CSV,%d,%d,%d,%d,%d,%lld,%.3f,%.3f,%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%lld\n",
                       world_size, actual_threads, rk, t,
                       (int)T[0], (long long)T[1],
                       T[2] * 1000.0, T[3] * 1000.0, (long long)T[4],
                       R[0] * 1000.0, R[1] * 1000.0, R[2] * 1000.0,
                       R[3] * 1000.0, R[4] * 1000.0, R[5] * 1000.0,
                       T[5] * 1000.0, (long long)R[6]);
            }
        }
        fflush(stdout);
        free(th_all);
        free(rk_all);
    }

    free(th_local);
    free(t_compute);
    free(t_idle);
    free(t_sched);
    free(t_rows);
    free(t_cpu);
    free(t_migr);

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include "hash.h"

inline void merge_keep_low(const float* __restrict a, int na,
                           const float* __restrict b, int nb,
                           float* out) {
    int i = 0, j = 0, k = 0;
    while (k < na && i < na && j < nb) {
        bool take_a = (a[i] <= b[j]);
        out[k++] = take_a ? a[i] : b[j];
        i += take_a;
        j += !take_a;
    }
    while (k < na && i < na) out[k++] = a[i++];
    while (k < na && j < nb) out[k++] = b[j++];
}

inline void merge_keep_high(const float* __restrict a, int na,
                            const float* __restrict b, int nb,
                            float* out) {
    int i = na - 1, j = nb - 1, k = na - 1;
    while (k >= 0 && i >= 0 && j >= 0) {
        bool take_a = (a[i] > b[j]);
        out[k--] = take_a ? a[i--] : b[j--];
    }
    while (k >= 0 && i >= 0) out[k--] = a[i--];
    while (k >= 0 && j >= 0) out[k--] = b[j--];
}

struct Bounds {
    float min;
    float max;
};

struct Segment {
    float start;
    float end;
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto fail = [&](const char *msg) {
        if (rank == 0) std::fprintf(stderr, "%s\n", msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
    };

    auto check = [&](int err) {
        if (err != MPI_SUCCESS) fail("MPI operation failed");
    };

    if (argc != 4 && argc != 5)
        fail("Usage: ./hw1 N input output [rounds=25]");

    auto positive = [&](const char *s) {
        char *end = nullptr;
        long long value = std::strtoll(s, &end, 10);
        if (!*s || *end || value < 1 || value > INT_MAX)
            fail("Invalid positive integer");
        return static_cast<int>(value);
    };

    const int n = positive(argv[1]);
    const int rounds = (argc == 5) ? positive(argv[4]) : 25;

    const int base = n / size;
    const int remainder = n % size;
    const int local_n = base + (rank < remainder);
    int global_start = rank * base + std::min(rank, remainder);

    const int max_partner_n = base + (remainder > 0 ? 1 : 0);
    std::vector<float> data(local_n);
    std::vector<float> incoming(max_partner_n);
    std::vector<float> merged(local_n);

    const int CHUNK = 32768;
    const int max_nch = (max_partner_n + CHUNK - 1) / CHUNK + 1;
    std::vector<MPI_Request> sreq(max_nch);
    std::vector<MPI_Request> rreq(max_nch);

    double io_time = 0.0;
    double sorting_time = 0.0;
    double merge_time = 0.0;
    double mpi_comm_time = 0.0;
    double hash_time = 0.0;

    double start_total = MPI_Wtime();

    float dummy_io = 0.0f;

    MPI_Info info;
    check(MPI_Info_create(&info));
    check(MPI_Info_set(info, "romio_cb_read", "enable"));
    check(MPI_Info_set(info, "romio_cb_write", "enable"));
    check(MPI_Info_set(info, "romio_no_indep_rw", "true"));
    check(MPI_Info_set(info, "cb_buffer_size", "16777216"));

    MPI_File file;
    check(MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, info, &file));

    MPI_Offset bytes;
    check(MPI_File_get_size(file, &bytes));
    if (bytes != static_cast<MPI_Offset>(n) * static_cast<MPI_Offset>(sizeof(float)))
        fail("Input size does not match N");

    double t_io_start = MPI_Wtime();
    check(MPI_File_read_at_all(file, static_cast<MPI_Offset>(global_start) * sizeof(float),
                               (local_n > 0 ? data.data() : &dummy_io), local_n, MPI_FLOAT, MPI_STATUS_IGNORE));
    io_time += MPI_Wtime() - t_io_start;
    check(MPI_File_close(&file));

    for (float value : data) {
        if (!std::isfinite(value)) fail("Input must contain finite floats");
    }

    for (int round = 0; round < rounds; ++round) {

        if (local_n > 0) {
            double t_sort_start = MPI_Wtime();
            boost::sort::spreadsort::float_sort(data.begin(), data.end());
            sorting_time += MPI_Wtime() - t_sort_start;
        }

        for (int iter = 0; iter < size; ++iter) {
            int local_swapped = 0;

            for (int phase = 0; phase < size + 1; ++phase) {
                bool is_even_phase = (phase % 2 == 0);
                int partner = -1;

                if (is_even_phase) {
                    if (rank % 2 == 0) partner = rank + 1;
                    else partner = rank - 1;
                } else {
                    if (rank % 2 == 1) partner = rank + 1;
                    else partner = rank - 1;
                }

                int peer = (partner < 0 || partner >= size) ? MPI_PROC_NULL : partner;

                if (peer != MPI_PROC_NULL) {
                    Bounds my_bound{ (local_n > 0) ? data.front() : INFINITY,
                                     (local_n > 0) ? data.back()  : -INFINITY };
                    Bounds partner_bound;
                    int preflight_tag = 1000 + phase;

                    MPI_Request reqs[2];
                    double t_comm_start = MPI_Wtime();
                    check(MPI_Irecv(&partner_bound, 2, MPI_FLOAT, peer, preflight_tag, MPI_COMM_WORLD, &reqs[0]));
                    check(MPI_Isend(&my_bound, 2, MPI_FLOAT, peer, preflight_tag, MPI_COMM_WORLD, &reqs[1]));
                    check(MPI_Wait(&reqs[0], MPI_STATUS_IGNORE));
                    mpi_comm_time += MPI_Wtime() - t_comm_start;

                    bool left_keeps_low = (rank < peer);
                    bool ordered = left_keeps_low ? (my_bound.max <= partner_bound.min)
                                                  : (partner_bound.max <= my_bound.min);

                    t_comm_start = MPI_Wtime();
                    check(MPI_Wait(&reqs[1], MPI_STATUS_IGNORE));
                    mpi_comm_time += MPI_Wtime() - t_comm_start;

                    if (ordered) {
                        continue;
                    }

                    local_swapped = 1;
                    const int partner_n = base + (partner < remainder);
                    
                    //find the number of elements to send to the partner
                    int nA = left_keeps_low ? local_n : partner_n;
                    int nB = left_keeps_low ? partner_n : local_n;
                    int low = std::max(0, nA - nB);
                    int high = std::min(nA, nB);
                    int send_cnt = high;

                    MPI_Request coranking_reqs[2];
                    Segment local_segment, remote_segment;
                    int coranking_tag = 2000 + phase;
                    check(MPI_Recv_init(&remote_segment, 2, MPI_FLOAT, peer, coranking_tag, MPI_COMM_WORLD, &coranking_reqs[0]));
                    check(MPI_Send_init(&local_segment, 2, MPI_FLOAT, peer, coranking_tag, MPI_COMM_WORLD, &coranking_reqs[1]));

                    while (low <= high) {
                        int partner_i = (low + high) >> 1;
                        int local_i = nA - partner_i;

                        local_segment = left_keeps_low
                            ? Segment{ (local_i > 0 && local_i <= local_n) ? data[local_i - 1] : -INFINITY,
                                       (local_i >= 0 && local_i < local_n) ? data[local_i] : INFINITY }
                            : Segment{ (partner_i > 0 && partner_i <= local_n) ? data[partner_i - 1] : -INFINITY,
                                       (partner_i >= 0 && partner_i < local_n) ? data[partner_i] : INFINITY };
                        remote_segment = { -INFINITY, +INFINITY };

                        t_comm_start = MPI_Wtime();
                        check(MPI_Startall(2, coranking_reqs));
                        check(MPI_Wait(&coranking_reqs[0], MPI_STATUS_IGNORE));
                        mpi_comm_time += MPI_Wtime() - t_comm_start;

                        Segment *lhs = left_keeps_low ? &local_segment : &remote_segment;
                        Segment *rhs = left_keeps_low ? &remote_segment : &local_segment;

                        if ((lhs->start <= rhs->end) && (rhs->start <= lhs->end)) {
                            send_cnt = partner_i;
                            high = partner_i - 1;
                        } else if (lhs->start > rhs->end) {
                            low = partner_i + 1;
                        } else {
                            high = partner_i - 1;
                        }
                        t_comm_start = MPI_Wtime();
                        check(MPI_Wait(&coranking_reqs[1], MPI_STATUS_IGNORE));
                        mpi_comm_time += MPI_Wtime() - t_comm_start;
                    }
                    MPI_Request_free(&coranking_reqs[0]);
                    MPI_Request_free(&coranking_reqs[1]);

                    const int transfer_stage_tag = 3000 + phase;
                    const float* send_ptr = left_keeps_low ? (data.data() + (local_n - send_cnt)) : data.data();
                    const int NCH = (send_cnt + CHUNK - 1) / CHUNK;

                    if (send_cnt <= CHUNK || NCH <= 1) {
                        t_comm_start = MPI_Wtime();
                        check(MPI_Irecv(incoming.data(), send_cnt, MPI_FLOAT, peer, transfer_stage_tag, MPI_COMM_WORLD, &reqs[0]));
                        check(MPI_Isend(send_ptr, send_cnt, MPI_FLOAT, peer, transfer_stage_tag, MPI_COMM_WORLD, &reqs[1]));
                        check(MPI_Wait(&reqs[0], MPI_STATUS_IGNORE));
                        mpi_comm_time += MPI_Wtime() - t_comm_start;

                        double t_merge_start = MPI_Wtime();
                        if (left_keeps_low) merge_keep_low(data.data(), local_n, incoming.data(), send_cnt, merged.data());
                        else                merge_keep_high(data.data(), local_n, incoming.data(), send_cnt, merged.data());
                        merge_time += MPI_Wtime() - t_merge_start;

                        t_comm_start = MPI_Wtime();
                        check(MPI_Wait(&reqs[1], MPI_STATUS_IGNORE));
                        mpi_comm_time += MPI_Wtime() - t_comm_start;
                    } else {
                        t_comm_start = MPI_Wtime();
                        for (int c = 0; c < NCH; ++c) {
                            const int off = c * CHUNK;
                            const int cnt = (off + CHUNK <= send_cnt) ? CHUNK : (send_cnt - off);
                            check(MPI_Irecv(incoming.data() + off, cnt, MPI_FLOAT, peer, transfer_stage_tag + c, MPI_COMM_WORLD, &rreq[c]));
                            check(MPI_Isend(send_ptr + off, cnt, MPI_FLOAT, peer, transfer_stage_tag + c, MPI_COMM_WORLD, &sreq[c]));
                        }
                        mpi_comm_time += MPI_Wtime() - t_comm_start;

                        double t_merge_start = MPI_Wtime();
                        if (left_keeps_low) {
                            int out = 0, i = 0;
                            for (int c = 0; c < NCH && out < local_n; ++c) {
                                const int off = c * CHUNK;
                                const int cnt = (off + CHUNK <= send_cnt) ? CHUNK : (send_cnt - off);

                                double tc = MPI_Wtime();
                                check(MPI_Wait(&rreq[c], MPI_STATUS_IGNORE));
                                mpi_comm_time += MPI_Wtime() - tc;

                                float* pb = incoming.data() + off;
                                int j = 0;
                                while (out < local_n && i < local_n && j < cnt) {
                                    bool take_a = (data[i] <= pb[j]);
                                    merged[out++] = take_a ? data[i++] : pb[j++];
                                }
                                while (out < local_n && i >= local_n && j < cnt) {
                                    merged[out++] = pb[j++];
                                }
                            }
                            while (out < local_n && i < local_n) merged[out++] = data[i++];
                        } else {
                            int out = 0, i = local_n - 1;
                            for (int c = NCH - 1; c >= 0 && out < local_n; --c) {
                                const int off = c * CHUNK;
                                const int cnt = (off + CHUNK <= send_cnt) ? CHUNK : (send_cnt - off);

                                double tc = MPI_Wtime();
                                check(MPI_Wait(&rreq[c], MPI_STATUS_IGNORE));
                                mpi_comm_time += MPI_Wtime() - tc;

                                float* pb = incoming.data() + off;
                                int j = cnt - 1;
                                while (out < local_n && i >= 0 && j >= 0) {
                                    bool take_a = (data[i] >= pb[j]);
                                    merged[local_n - 1 - out++] = take_a ? data[i--] : pb[j--];
                                }
                                while (out < local_n && i < 0 && j >= 0) {
                                    merged[local_n - 1 - out++] = pb[j--];
                                }
                            }
                            while (out < local_n && i >= 0) merged[local_n - 1 - out++] = data[i--];
                        }
                        merge_time += MPI_Wtime() - t_merge_start;

                        t_comm_start = MPI_Wtime();
                        check(MPI_Waitall(NCH, sreq.data(), MPI_STATUSES_IGNORE));
                        mpi_comm_time += MPI_Wtime() - t_comm_start;
                    }

                    std::swap(data, merged);
                }
            }

            int global_swapped = 0;
            double t_comm_start = MPI_Wtime();
            check(MPI_Allreduce(&local_swapped, &global_swapped, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD));
            mpi_comm_time += MPI_Wtime() - t_comm_start;

            if (global_swapped == 0) {
                break;
            }
        }

        int prefix = 0;
        double t_comm_start = MPI_Wtime();
        check(MPI_Exscan(&local_n, &prefix, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
        if (rank == 0) prefix = 0;
        global_start = prefix;
        mpi_comm_time += MPI_Wtime() - t_comm_start;

        if (round + 1 < rounds) {
            double t_hash_start = MPI_Wtime();
            for (int i = 0; i < local_n; ++i) {
                data[i] = hw1_hash(data[i], global_start + i, round + 1);
            }
            hash_time += MPI_Wtime() - t_hash_start;
        }
    }

    t_io_start = MPI_Wtime();
    check(MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY,
                        info, &file));
    check(MPI_File_set_size(file, static_cast<MPI_Offset>(n) * static_cast<MPI_Offset>(sizeof(float))));
    check(MPI_File_write_at_all(file, static_cast<MPI_Offset>(global_start) * sizeof(float),
                                (local_n > 0 ? data.data() : &dummy_io), local_n, MPI_FLOAT, MPI_STATUS_IGNORE));
    check(MPI_File_close(&file));
    MPI_Info_free(&info);
    io_time += MPI_Wtime() - t_io_start;

    double total_time = MPI_Wtime() - start_total;

    double max_io, max_sort, max_merge, max_comm, max_hash, max_total;
    MPI_Reduce(&io_time, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sorting_time, &max_sort, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&merge_time, &max_merge, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mpi_comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hash_time, &max_hash, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("\n=========================================\n");
        std::printf("     HW1 Performance Profiling (Max)\n");
        std::printf("=========================================\n");
        std::printf(" I/O Time      : %.6f seconds\n", max_io);
        std::printf(" Sorting Time  : %.6f seconds\n", max_sort);
        std::printf(" Merge Time    : %.6f seconds\n", max_merge);
        std::printf(" MPI Comm Time : %.6f seconds\n", max_comm);
        std::printf(" Hash Time     : %.6f seconds\n", max_hash);
        std::printf("-----------------------------------------\n");
        std::printf(" Total Execution Time: %.6f seconds\n", max_total);
        std::printf("=========================================\n");
    }

    MPI_Finalize();
    return 0;
}
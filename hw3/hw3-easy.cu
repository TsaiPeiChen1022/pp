#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

const int INF = (1 << 30) - 1;

// 旗標位元定義
constexpr int FLAG_IN_PEND        = 1 << 0; // 1: 已在 pending 佇列
constexpr int FLAG_IN_NEXT_LIGHT  = 1 << 1; // 2: 已在下一輪 light_frontier 佇列
constexpr int FLAG_IN_BUCKET_DONE = 1 << 2; // 4: 已加入本桶子的 bucket_done 集合

struct Edge {
    int to;
    int w;
};

// TODO: add kernels: init distances, relax frontier edges (atomicMin / atomicAdd), find next bucket

// [新增 Block-per-Source] 每個 Block 獨立負責多個起點（動態搶號），完整跑完單一 src 的 Delta-Stepping
// T: 每個 Block 的執行緒數；G: 幾個執行緒一組（sub-warp）共同走訪同一個頂點的鄰接邊
template <int T, int G>
__global__ void __launch_bounds__(T)
apsp_block_per_src_kernel(const int* __restrict__ light_row_ptr,
                          const Edge* __restrict__ light_adj,
                          const int* __restrict__ heavy_row_ptr,
                          const Edge* __restrict__ heavy_adj,
                          int n, int delta,
                          int src_base, int src_count,
                          int* __restrict__ d_out_dist,
                          int* __restrict__ d_block_state,
                          int* __restrict__ d_src_counter) {
    static_assert(T % G == 0, "Block threads T must be divisible by group size G");

    // 區塊內共用控制變數（全部放在 Shared Memory，避免迴圈內頻繁回 Host 同步）
    __shared__ int s_sidx;
    __shared__ int s_min_b;
    __shared__ int s_cb;
    __shared__ int s_pend_cnt;
    __shared__ int s_next_pend_cnt;
    __shared__ int s_cur_cnt;
    __shared__ int s_nxt_cnt;
    __shared__ int s_R_cnt;

    const int tid = threadIdx.x;
    const int grp = tid / G;
    const int lane = tid % G;
    constexpr int ngrp = T / G;

    // 每個 Block 獨佔 6 * n 個 int 的工作區：
    // [0..n-1] flags, [n..2n-1] pendA, [2n..3n-1] pendB, [3n..4n-1] cur, [4n..5n-1] nxt, [5n..6n-1] R
    int* base      = d_block_state + (size_t)blockIdx.x * 6ULL * (size_t)n;
    int* flags     = base;
    int* pendA     = base + (size_t)n;
    int* pendB     = base + 2ULL * (size_t)n;
    int* cur       = base + 3ULL * (size_t)n;
    int* nxt       = base + 4ULL * (size_t)n;
    int* R         = base + 5ULL * (size_t)n;

    while (true) {
        if (tid == 0) {
            s_sidx = atomicAdd(d_src_counter, 1);
        }
        __syncthreads();

        const int sidx = s_sidx;
        if (sidx >= src_count) break;

        const int src = src_base + sidx;
        int* dist = d_out_dist + (size_t)sidx * (size_t)n;

        // 初始化此 src 的距離與旗標
        for (int v = tid; v < n; v += T) {
            if (v == src) {
                dist[v] = 0;
                flags[v] = FLAG_IN_PEND;
            } else {
                dist[v] = INF;
                flags[v] = 0;
            }
        }

        if (tid == 0) {
            pendA[0] = src;
            s_pend_cnt = 1;
            s_next_pend_cnt = 0;
            s_cur_cnt = 0;
            s_nxt_cnt = 0;
            s_R_cnt = 0;
            s_cb = -1;
        }
        __syncthreads();

        int* pend = pendA;
        int* next_pend = pendB;

        while (true) {
            // 1. 尋找目前 pending 中、桶號 > s_cb 的最小桶號
            if (tid == 0) {
                s_min_b = INT_MAX;
                s_R_cnt = 0;
                s_cur_cnt = 0;
                s_next_pend_cnt = 0;
            }
            __syncthreads();

            const int pc = s_pend_cnt;
            const int cb = s_cb;

            for (int i = tid; i < pc; i += T) {
                int v = pend[i];
                int dv = dist[v];
                if (dv < INF) {
                    int b = dv / delta;
                    if (b > cb) {
                        atomicMin(&s_min_b, b);
                    }
                }
            }
            __syncthreads();

            const int min_b = s_min_b;
            if (min_b == INT_MAX) {
                // 清理殘留 FLAG_IN_PEND 後結束此 src
                for (int i = tid; i < pc; i += T) {
                    atomicAnd(&flags[pend[i]], ~FLAG_IN_PEND);
                }
                __syncthreads();
                break;
            }

            // 2. 分派 pending 頂點：屬於 min_b 進入 cur 與 R；大於 min_b 留到 next_pend
            for (int i = tid; i < pc; i += T) {
                int v = pend[i];
                atomicAnd(&flags[v], ~FLAG_IN_PEND);
                int dv = dist[v];
                if (dv < INF) {
                    int b = dv / delta;
                    if (b == min_b) {
                        int cpos = atomicAdd(&s_cur_cnt, 1);
                        cur[cpos] = v;
                        if ((atomicOr(&flags[v], FLAG_IN_BUCKET_DONE) & FLAG_IN_BUCKET_DONE) == 0) {
                            int rpos = atomicAdd(&s_R_cnt, 1);
                            R[rpos] = v;
                        }
                    } else if (b > min_b) {
                        if ((atomicOr(&flags[v], FLAG_IN_PEND) & FLAG_IN_PEND) == 0) {
                            int ppos = atomicAdd(&s_next_pend_cnt, 1);
                            next_pend[ppos] = v;
                        }
                    }
                }
            }
            __syncthreads();

            if (tid == 0) {
                s_pend_cnt = s_next_pend_cnt;
                s_cb = min_b;
                s_nxt_cnt = 0;
            }
            {
                int* tmp = pend;
                pend = next_pend;
                next_pend = tmp;
            }
            __syncthreads();

            const long long bound = (long long)(min_b + 1) * (long long)delta;

            // 3. Phase 1：同桶內輕邊閉包迭代（由 G 個 thread 協作掃描單一頂點的輕邊）
            while (s_cur_cnt > 0) {
                const int cc = s_cur_cnt;
                for (int i = grp; i < cc; i += ngrp) {
                    int v = cur[i];
                    int dv = dist[v];
                    if (dv >= INF) continue;

                    int e_begin = light_row_ptr[v];
                    int e_end = light_row_ptr[v + 1];
                    for (int e = e_begin + lane; e < e_end; e += G) {
                        Edge edge = light_adj[e];
                        long long nd_ll = (long long)dv + (long long)edge.w;
                        if (nd_ll >= INF) continue;
                        int nd = (int)nd_ll;
                        int u = edge.to;

                        if (nd < dist[u]) {
                            int old = atomicMin(&dist[u], nd);
                            if (nd < old) {
                                if ((long long)nd < bound) {
                                    if ((flags[u] & FLAG_IN_NEXT_LIGHT) == 0) {
                                        if ((atomicOr(&flags[u], FLAG_IN_NEXT_LIGHT) & FLAG_IN_NEXT_LIGHT) == 0) {
                                            int pos = atomicAdd(&s_nxt_cnt, 1);
                                            nxt[pos] = u;
                                        }
                                    }
                                } else {
                                    if ((flags[u] & FLAG_IN_PEND) == 0) {
                                        if ((atomicOr(&flags[u], FLAG_IN_PEND) & FLAG_IN_PEND) == 0) {
                                            int pos = atomicAdd(&s_pend_cnt, 1);
                                            pend[pos] = u;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();

                // 將 nxt 推進為下一輪 cur，並去重併入 R
                const int nn = s_nxt_cnt;
                for (int i = tid; i < nn; i += T) {
                    int v = nxt[i];
                    cur[i] = v;
                    atomicAnd(&flags[v], ~FLAG_IN_NEXT_LIGHT);
                    if ((atomicOr(&flags[v], FLAG_IN_BUCKET_DONE) & FLAG_IN_BUCKET_DONE) == 0) {
                        int rpos = atomicAdd(&s_R_cnt, 1);
                        R[rpos] = v;
                    }
                }
                __syncthreads();

                if (tid == 0) {
                    s_cur_cnt = nn;
                    s_nxt_cnt = 0;
                }
                __syncthreads();
            }

            // 4. Phase 2：對本桶所有累積頂點 R 鬆弛重邊
            const int rc = s_R_cnt;
            if (rc > 0) {
                for (int i = grp; i < rc; i += ngrp) {
                    int v = R[i];
                    int dv = dist[v];
                    if (dv >= INF) continue;

                    int e_begin = heavy_row_ptr[v];
                    int e_end = heavy_row_ptr[v + 1];
                    for (int e = e_begin + lane; e < e_end; e += G) {
                        Edge edge = heavy_adj[e];
                        long long nd_ll = (long long)dv + (long long)edge.w;
                        if (nd_ll >= INF) continue;
                        int nd = (int)nd_ll;
                        int u = edge.to;

                        if (nd < dist[u]) {
                            int old = atomicMin(&dist[u], nd);
                            if (nd < old) {
                                if ((flags[u] & FLAG_IN_PEND) == 0) {
                                    if ((atomicOr(&flags[u], FLAG_IN_PEND) & FLAG_IN_PEND) == 0) {
                                        int pos = atomicAdd(&s_pend_cnt, 1);
                                        pend[pos] = u;
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();

                for (int i = tid; i < rc; i += T) {
                    atomicAnd(&flags[R[i]], ~FLAG_IN_BUCKET_DONE);
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    // TODO: get device properties
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // Read input: n, m, then m edges (src, dst, weight)
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 1; }
    int n, m;
    if (fread(&n, sizeof(int), 1, fin) != 1 || fread(&m, sizeof(int), 1, fin) != 1) {
        fprintf(stderr, "bad input header\n");
        return 1;
    }
    std::vector<int> raw((size_t)m * 3);
    if (fread(raw.data(), sizeof(int), raw.size(), fin) != raw.size()) {
        fprintf(stderr, "truncated input\n");
        return 1;
    }
    fclose(fin);

    // Bucket width: same default as hw3-1.cu
    long long wsum = 0;
    for (int i = 0; i < m; ++i) wsum += raw[3 * i + 2];
    const double avg_w = m > 0 ? (double)wsum / m : 1.0;
    const double avg_deg = n > 0 ? std::max(1.0, (double)m / n) : 1.0;
    int delta = std::max(1, (int)(avg_w * 32.0 / avg_deg));
    if (const char* s = getenv("HW3_DELTA")) delta = std::max(1, atoi(s));

    // 建立兩份 CSR：輕邊（w <= delta）與重邊（w > delta）分開存放
    std::vector<int> light_row_ptr(n + 1, 0), heavy_row_ptr(n + 1, 0);
    for (int i = 0; i < m; ++i) {
        const int src = raw[3 * i];
        const int w = raw[3 * i + 2];
        if (w <= delta) {
            light_row_ptr[src + 1]++;
        } else {
            heavy_row_ptr[src + 1]++;
        }
    }
    for (int v = 0; v < n; ++v) {
        light_row_ptr[v + 1] += light_row_ptr[v];
        heavy_row_ptr[v + 1] += heavy_row_ptr[v];
    }

    const int light_m = light_row_ptr[n];
    const int heavy_m = heavy_row_ptr[n];
    std::vector<Edge> light_adj(std::max(light_m, 1));
    std::vector<Edge> heavy_adj(std::max(heavy_m, 1));
    std::vector<int> light_next(light_row_ptr.begin(), light_row_ptr.end() - 1);
    std::vector<int> heavy_next(heavy_row_ptr.begin(), heavy_row_ptr.end() - 1);

    for (int i = 0; i < m; ++i) {
        const int src = raw[3 * i];
        const int dst = raw[3 * i + 1];
        const int w = raw[3 * i + 2];
        if (w <= delta) {
            light_adj[light_next[src]++] = Edge{dst, w};
        } else {
            heavy_adj[heavy_next[src]++] = Edge{dst, w};
        }
    }

    // TODO: cudaMalloc device arrays, cudaMemcpy graph to device
    // ---- Device 端共用輕／重邊 CSR 宣告與配置 ----
    int *d_light_row_ptr = nullptr, *d_heavy_row_ptr = nullptr;
    Edge *d_light_adj = nullptr, *d_heavy_adj = nullptr;

    cudaMalloc(&d_light_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&d_heavy_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&d_light_adj, (size_t)std::max(light_m, 1) * sizeof(Edge));
    cudaMalloc(&d_heavy_adj, (size_t)std::max(heavy_m, 1) * sizeof(Edge));

    cudaMemcpy(d_light_row_ptr, light_row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heavy_row_ptr, heavy_row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (light_m > 0) {
        cudaMemcpy(d_light_adj, light_adj.data(), (size_t)light_m * sizeof(Edge), cudaMemcpyHostToDevice);
    }
    if (heavy_m > 0) {
        cudaMemcpy(d_heavy_adj, heavy_adj.data(), (size_t)heavy_m * sizeof(Edge), cudaMemcpyHostToDevice);
    }

    // [新增 Block-per-Source] 依可用顯存決定每批處理的起點數與常駐 Block 數
    constexpr int TPB = 256;
    constexpr int GRP = 4;

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    // 每個起點輸出 n 個 int，每個常駐 Block 佔用 6 * n 個 int
    const size_t per_src_bytes = std::max((size_t)1, (size_t)n * sizeof(int));
    const size_t per_block_state_bytes = std::max((size_t)1, (size_t)6 * (size_t)n * sizeof(int));
    const size_t usable_bytes = (free_bytes > (64ULL << 20)) ? (free_bytes - (64ULL << 20)) : (free_bytes / 2);

    int num_blocks = std::max(1, std::min(n > 0 ? n : 1, prop.multiProcessorCount * 4));
    while (num_blocks > 1 && (size_t)num_blocks * per_block_state_bytes > usable_bytes / 2) {
        num_blocks /= 2;
    }

    size_t rem_for_out = (usable_bytes > (size_t)num_blocks * per_block_state_bytes)
                             ? (usable_bytes - (size_t)num_blocks * per_block_state_bytes)
                             : per_src_bytes;
    int batch_size = (n > 0) ? (int)std::max((size_t)1, std::min((size_t)n, rem_for_out / per_src_bytes)) : 0;

    int* d_out_dist = nullptr;
    int* d_block_state = nullptr;
    int* d_src_counter = nullptr;

    cudaMalloc(&d_out_dist, (size_t)std::max(batch_size, 1) * (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_block_state, (size_t)num_blocks * (size_t)6 * (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_src_counter, sizeof(int));

    std::vector<int> host_batch_dist((size_t)std::max(batch_size, 1) * (size_t)std::max(n, 1));

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    // TODO: replace whole loop with kernel launches
    // [新增 Block-per-Source] 分批啟動常駐 Block 搶算起點，批次完成後一次搬回 Host 寫檔
    for (int src_base = 0; src_base < n; src_base += batch_size) {
        const int cur_batch_count = std::min(batch_size, n - src_base);
        const int cur_blocks = std::min(num_blocks, cur_batch_count);

        cudaMemset(d_src_counter, 0, sizeof(int));
        apsp_block_per_src_kernel<TPB, GRP><<<cur_blocks, TPB>>>(
            d_light_row_ptr, d_light_adj,
            d_heavy_row_ptr, d_heavy_adj,
            n, delta,
            src_base, cur_batch_count,
            d_out_dist, d_block_state, d_src_counter
        );
        cudaDeviceSynchronize();

        // TODO: cudaMemcpy distances back to host
        const size_t batch_elems = (size_t)cur_batch_count * (size_t)n;
        cudaMemcpy(host_batch_dist.data(), d_out_dist, batch_elems * sizeof(int), cudaMemcpyDeviceToHost);

        if (fwrite(host_batch_dist.data(), sizeof(int), batch_elems, fout) != batch_elems) {
            perror("write output");
            fclose(fout);
            return 1;
        }
    }
    fclose(fout);

    // TODO: cudaFree device arrays
    // ---- 釋放 Device 記憶體 ----
    cudaFree(d_out_dist);
    cudaFree(d_block_state);
    cudaFree(d_src_counter);
    cudaFree(d_light_row_ptr);
    cudaFree(d_heavy_row_ptr);
    cudaFree(d_light_adj);
    cudaFree(d_heavy_adj);

    return 0;
}
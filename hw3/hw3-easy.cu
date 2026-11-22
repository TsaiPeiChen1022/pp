// 版本 v5 = v4 + [優化5] 上界剪枝（dv + w > U 就 break）
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <algorithm>
#include <chrono>
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

// [優化5] 上界剪枝用的輔助函式
__device__ __forceinline__ int warp_max(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// [優化5] 當「所有頂點都已到達」(s_reached == n) 後，U = 目前所有 tentative 距離的最大值。
//         tentative 距離只會變小，所以 U 永遠是所有最終距離的合法上界；
//         鬆弛邊時若 dv + w > U，該邊不可能改善任何頂點（dist[to] <= U），
//         而每個頂點的邊依權重排序，後面的邊只會更大 -> 可以直接 break。
//         有不可達頂點時 s_reached 永遠 < n，不會啟用剪枝，結果不受影響。
//         整個 Block 必須一起呼叫（內含 __syncthreads）。
template <int T>
__device__ __forceinline__ void update_bound(const int* dist, int n, int tid,
                                             const int* s_reached, int* s_U, int* s_doU,
                                             int& U, int& work, int thresh) {
    if (tid == 0) *s_doU = (*s_reached == n) && (U == INT_MAX || work >= thresh);
    __syncthreads();
    if (*s_doU) {
        if (tid == 0) *s_U = 0;
        __syncthreads();
        int lm = 0;
        for (int v = tid; v < n; v += T) lm = max(lm, dist[v]);
        lm = warp_max(lm);
        if ((tid & 31) == 0) atomicMax(s_U, lm);
        __syncthreads();
        U = *s_U;
        work = 0;
    }
}

// [新增 Block-per-Source] 每個 Block 獨立負責多個起點（動態搶號），完整跑完單一 src 的 Delta-Stepping
// T: 每個 Block 的執行緒數；G: 幾個執行緒一組（sub-warp）共同走訪同一個頂點的鄰接邊
template <int T, int G>
__global__ void __launch_bounds__(T)
apsp_block_per_src_kernel(const int* __restrict__ row_ptr,      // [優化4] 單一 CSR 的列起點
                          const int* __restrict__ mid,          // [優化4] 輕邊 / 重邊分界
                          const unsigned* __restrict__ adj,     // [優化4] 4 bytes 壓縮邊 (w<<16)|to
                          int n, int delta,
                          int prune,                            // [優化5] 是否啟用上界剪枝
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
    __shared__ int s_reached, s_U, s_doU;   // [優化5] 已到達頂點數 / 上界 U / 是否重算 U

    const int tid = threadIdx.x;
    const int grp = tid / G;
    const int lane = tid % G;
    constexpr int ngrp = T / G;
    const int thresh = max(512, n / 4);      // [優化5] 每處理約 n/4 個頂點才重算一次 U

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
            s_reached = 1;                       // [優化5] 起點自己算已到達
        }
        __syncthreads();

        int* pend = pendA;
        int* next_pend = pendB;
        int U = INT_MAX;                         // [優化5] INT_MAX = 尚未啟用剪枝
        int work = 0;

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
                if (prune) update_bound<T>(dist, n, tid, &s_reached, &s_U, &s_doU, U, work, thresh);  // [優化5]
                const int cc = s_cur_cnt;
                work += cc;
                for (int i = grp; i < cc; i += ngrp) {
                    int v = cur[i];
                    int dv = dist[v];
                    if (dv >= INF) continue;

                    int e_begin = row_ptr[v];       // [優化4] 輕邊範圍 [row_ptr[v], mid[v])
                    int e_end = mid[v];
                    for (int e = e_begin + lane; e < e_end; e += G) {
                        const unsigned pk = adj[e];     // [優化4] 解開壓縮邊
                        Edge edge = Edge{(int)(pk & 0xFFFFu), (int)(pk >> 16)};
                        long long nd_ll = (long long)dv + (long long)edge.w;
                        if (nd_ll > U) break;      // [優化5] 邊已排序，後面的邊只會更大，不可能再改善
                        if (nd_ll >= INF) continue;
                        int nd = (int)nd_ll;
                        int u = edge.to;

                        if (nd < dist[u]) {
                            int old = atomicMin(&dist[u], nd);
                            if (old == INF) atomicAdd(&s_reached, 1);   // [優化5] 第一次被到達
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
            if (prune) update_bound<T>(dist, n, tid, &s_reached, &s_U, &s_doU, U, work, thresh);      // [優化5]
            const int rc = s_R_cnt;
            if (rc > 0) {
                for (int i = grp; i < rc; i += ngrp) {
                    int v = R[i];
                    int dv = dist[v];
                    if (dv >= INF) continue;

                    int e_begin = mid[v];           // [優化4] 重邊範圍 [mid[v], row_ptr[v+1])
                    int e_end = row_ptr[v + 1];
                    for (int e = e_begin + lane; e < e_end; e += G) {
                        const unsigned pk = adj[e];     // [優化4] 解開壓縮邊
                        Edge edge = Edge{(int)(pk & 0xFFFFu), (int)(pk >> 16)};
                        long long nd_ll = (long long)dv + (long long)edge.w;
                        if (nd_ll > U) break;      // [優化5] 邊已排序，後面的邊只會更大，不可能再改善
                        if (nd_ll >= INF) continue;
                        int nd = (int)nd_ll;
                        int u = edge.to;

                        if (nd < dist[u]) {
                            int old = atomicMin(&dist[u], nd);
                            if (old == INF) atomicAdd(&s_reached, 1);   // [優化5] 第一次被到達
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

// ---- [優化3] GPU 建圖 kernel 開始 ----
// [優化4] 單一 CSR：先依 src 分組（count + scatter，和 v3 相同），再「一個 Block 負責一個頂點」
//         用 shared memory 的 counting sort 把該頂點的邊依權重由小到大排好（w <= 1000 < 1024 個 bin）。
//         邊同時壓成 4 bytes：(w << 16) | to（原本 Edge 8 bytes）。

// 第 1 趟：每個頂點的出度 + w 總和（取代 wsum_kernel）
__global__ void count_src_kernel(const int* __restrict__ raw, size_t m, int* __restrict__ cnt,
                                 unsigned long long* __restrict__ wsum) {
    unsigned long long local = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        atomicAdd(&cnt[raw[i * 3]], 1);
        local += (unsigned)raw[i * 3 + 2];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(wsum, local);
}

// 第 2 趟：依 src 分組 scatter，內容是壓縮邊 (w << 16) | dst（組內順序不拘）
__global__ void scatter_src_kernel(const int* __restrict__ raw, size_t m, int* __restrict__ cur,
                                   unsigned* __restrict__ tmp) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        const int pos = atomicAdd(&cur[raw[i * 3]], 1);
        tmp[pos] = ((unsigned)raw[i * 3 + 2] << 16) | (unsigned)raw[i * 3 + 1];
    }
}

// 第 3 趟：一個 Block (1024 threads) 負責一個頂點，依權重做 counting sort
__global__ void __launch_bounds__(1024)
sort_by_w_kernel(const int* __restrict__ row, const unsigned* __restrict__ tmp, unsigned* __restrict__ adj) {
    __shared__ int hist[1024];
    __shared__ int scan[1024];
    __shared__ int cur[1024];
    const int tid = threadIdx.x;
    const int v = blockIdx.x;
    const int b = row[v], e = row[v + 1];

    hist[tid] = 0;
    __syncthreads();
    for (int i = b + tid; i < e; i += 1024) atomicAdd(&hist[tmp[i] >> 16], 1);
    __syncthreads();

    const int val = hist[tid];
    scan[tid] = val;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {       // Hillis-Steele inclusive scan
        const int x = (tid >= off) ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += x;
        __syncthreads();
    }
    cur[tid] = scan[tid] - val;                      // exclusive prefix = 該權重的起始位置
    __syncthreads();

    for (int i = b + tid; i < e; i += 1024) {
        const unsigned p = tmp[i];
        const int pos = atomicAdd(&cur[p >> 16], 1);
        adj[b + pos] = p;
    }
}

// mid[v] = 頂點 v 第一條「重邊 (w > delta)」的位置；因為邊依權重排序，用二分搜尋即可
__global__ void mid_kernel(const int* __restrict__ row, const unsigned* __restrict__ adj,
                           int* __restrict__ mid, int n, int delta) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int lo = row[v], hi = row[v + 1];
    while (lo < hi) {
        int md = (lo + hi) >> 1;
        if ((int)(adj[md] >> 16) <= delta) lo = md + 1; else hi = md;
    }
    mid[v] = lo;
}
// ---- [優化3] GPU 建圖 kernel 結束 ----

int main(int argc, char* argv[]) {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    const auto t_start = Clock::now();

    // TODO: get device properties
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // ---- [優化3] GPU 建圖 開始 ----
    // [新增計時] 階段 1：讀檔 + 建 CSR（[優化3] 之後 CSR 在 GPU 上建）
    const auto t_io_csr_begin = Clock::now();

    // Read input: n, m, then m edges (src, dst, weight)
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 1; }
    int n, m;
    if (fread(&n, sizeof(int), 1, fin) != 1 || fread(&m, sizeof(int), 1, fin) != 1) {
        fprintf(stderr, "bad input header\n");
        fclose(fin);
        return 1;
    }
    if (n > 65535) {   // [優化3] src / dst 需塞進 16 bits（hw3-1 的 V 上限是 40000，不會觸發）
        fprintf(stderr, "n too large\n");
        fclose(fin);
        return 1;
    }

    // [優化3] 原本是 std::vector<int> raw(3m) 一次 fread，之後在 CPU 兩趟建 CSR。
    //         現在改成：每次 fread 約 12 MB 到 pinned staging，再 cudaMemcpyAsync 到 device 的 d_raw，
    //         讀檔與 H2D 重疊，CPU 不再保留整份 raw。
    int* d_raw = nullptr;
    if (m > 0) cudaMalloc(&d_raw, (size_t)m * 3 * sizeof(int));
    {
        const size_t CHE = (size_t)1 << 20;   // 每段 1M 條邊
        int* stage[2];
        cudaEvent_t ev[2];
        for (int i = 0; i < 2; ++i) {
            cudaMallocHost(&stage[i], CHE * 3 * sizeof(int));
            cudaEventCreate(&ev[i]);
            cudaEventRecord(ev[i], 0);
        }
        size_t it = 0;
        for (size_t e0 = 0; e0 < (size_t)m; e0 += CHE, ++it) {
            const size_t cnt = std::min(CHE, (size_t)m - e0);
            const int k = (int)(it & 1);
            cudaEventSynchronize(ev[k]);      // 這個 staging buffer 上一次的 copy 已完成
            if (fread(stage[k], sizeof(int), cnt * 3, fin) != cnt * 3) {
                fprintf(stderr, "truncated input\n");
                fclose(fin);
                return 1;
            }
            cudaMemcpyAsync(d_raw + e0 * 3, stage[k], cnt * 3 * sizeof(int), cudaMemcpyHostToDevice, 0);
            cudaEventRecord(ev[k], 0);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < 2; ++i) { cudaFreeHost(stage[i]); cudaEventDestroy(ev[i]); }
    }
    fclose(fin);

    // [優化4] 改成「單一 CSR」：d_row_ptr / d_adj（每個頂點的邊依權重排序、4 bytes 壓縮）
    //         加上 d_mid（輕邊 / 重邊的分界）。原本的 light_*、heavy_* 兩份 row_ptr 與 Edge 陣列都不需要了。
    // TODO: cudaMalloc device arrays, cudaMemcpy graph to device
    int *d_row_ptr = nullptr, *d_mid = nullptr;
    unsigned* d_adj = nullptr;
    cudaMalloc(&d_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&d_mid, (size_t)std::max(n, 1) * sizeof(int));

    unsigned long long* d_wsum = nullptr;
    cudaMalloc(&d_wsum, sizeof(unsigned long long));
    cudaMemset(d_wsum, 0, sizeof(unsigned long long));
    const int kb = (int)std::max<size_t>(1, std::min<size_t>(((size_t)m + 255) / 256, 8192));

    std::vector<int> row_ptr(n + 1, 0);
    if (m > 0) {
        int* d_cnt = nullptr;                                   // 先當出度計數器，之後當 scatter 游標
        cudaMalloc(&d_cnt, (size_t)n * sizeof(int));
        cudaMemset(d_cnt, 0, (size_t)n * sizeof(int));
        count_src_kernel<<<kb, 256>>>(d_raw, (size_t)m, d_cnt, d_wsum);

        std::vector<int> h_cnt(n);
        cudaMemcpy(h_cnt.data(), d_cnt, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int v = 0; v < n; ++v) row_ptr[v + 1] = row_ptr[v] + h_cnt[v];   // prefix sum
        cudaMemcpy(d_row_ptr, row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cnt, row_ptr.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice);  // 游標 = row_ptr

        unsigned* d_tmp = nullptr;
        cudaMalloc(&d_tmp, (size_t)m * sizeof(unsigned));
        cudaMalloc(&d_adj, (size_t)m * sizeof(unsigned));
        scatter_src_kernel<<<kb, 256>>>(d_raw, (size_t)m, d_cnt, d_tmp);
        cudaFree(d_raw);
        d_raw = nullptr;

        sort_by_w_kernel<<<n, 1024>>>(d_row_ptr, d_tmp, d_adj);  // 每個頂點的邊依權重排序
        cudaDeviceSynchronize();
        cudaFree(d_tmp);
        cudaFree(d_cnt);
    } else {
        cudaMalloc(&d_adj, sizeof(unsigned));
        cudaMemset(d_row_ptr, 0, (size_t)(n + 1) * sizeof(int));
    }

    unsigned long long h_wsum = 0;
    cudaMemcpy(&h_wsum, d_wsum, sizeof(h_wsum), cudaMemcpyDeviceToHost);
    cudaFree(d_wsum);

    // Bucket width: same default as hw3-1.cu
    const long long wsum = (long long)h_wsum;
    const double avg_w = m > 0 ? (double)wsum / m : 1.0;
    const double avg_deg = n > 0 ? std::max(1.0, (double)m / n) : 1.0;
    int delta = std::max(1, (int)(avg_w * 32.0 / avg_deg));
    if (const char* s = getenv("HW3_DELTA")) delta = std::max(1, atoi(s));

    // [優化5] 稠密圖 (平均出度 >= 16) 才啟用上界剪枝；可用 HW3_PRUNE=0/1 強制關閉/開啟
    int prune = avg_deg >= 16 ? 1 : 0;
    if (const char* s = getenv("HW3_PRUNE")) prune = atoi(s) ? 1 : 0;

    // [優化4] delta 決定後才算 mid[v]
    mid_kernel<<<(n + 255) / 256, 256>>>(d_row_ptr, d_adj, d_mid, n, delta);

    cudaDeviceSynchronize();
    const double read_and_csr_ms = Ms(Clock::now() - t_io_csr_begin).count();
    // ---- [優化3] GPU 建圖 結束 ----

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

    // [優化2] 改用 2 組 stream 輪流：GPU 算下一批時，Host 同時 fwrite 上一批。
    //         每個 stream 各自擁有 d_out_dist / d_block_state / pinned buffer，
    //         為了和 v1 的「總常駐 Block 數」相同，每個 stream 只啟動 num_blocks / 2 個 Block。
    const int num_blocks_per = std::max(1, num_blocks / 2);

    // [優化1] 原本 batch_size 只受「剩餘顯存」限制，會讓 d_out_dist 和 pinned buffer 變成數 GB，
    //         nsys timeline 上 cudaHostAlloc 約佔 1.5 s。改成每批輸出上限約 128 MB，
    //         pinned buffer 因此只剩 128 MB，配置時間幾乎可忽略。
    {
        const size_t chunk_bytes = (size_t)128 << 20;
        const size_t cap = std::max((size_t)num_blocks_per * 3, chunk_bytes / per_src_bytes);  // 至少讓每個 Block 有 3 個 src 可做
        batch_size = (int)std::max((size_t)1, std::min((size_t)batch_size, cap));
    }

    const size_t max_batch_elems = (size_t)std::max(batch_size, 1) * (size_t)std::max(n, 1);

    // [優化2] 兩組 (stream / d_out_dist / d_block_state / pinned buffer) 輪流使用
    int* d_out_dist[2] = {nullptr, nullptr};
    int* d_block_state[2] = {nullptr, nullptr};
    int* d_src_counter = nullptr;               // 兩個 stream 各用一個計數器 (d_src_counter + t)
    int* host_batch_dist[2] = {nullptr, nullptr}; // [修改 Pinned] 改用 cudaMallocHost 配置 pinned 緩衝區
    cudaStream_t stream[2];

    cudaMalloc(&d_src_counter, 2 * sizeof(int));
    for (int t = 0; t < 2; ++t) {
        cudaStreamCreate(&stream[t]);
        cudaMalloc(&d_out_dist[t], max_batch_elems * sizeof(int));
        cudaMalloc(&d_block_state[t], (size_t)num_blocks_per * (size_t)6 * (size_t)std::max(n, 1) * sizeof(int));
        if (cudaMallocHost(&host_batch_dist[t], max_batch_elems * sizeof(int)) != cudaSuccess || !host_batch_dist[t]) {
            fprintf(stderr, "cudaMallocHost failed\n");
            return 1;
        }
    }

    // [優化2] 所有 Device / Host 資源集中在這裡釋放（錯誤路徑與正常結束共用）
    auto release_all = [&]() {
        for (int t = 0; t < 2; ++t) {
            cudaFreeHost(host_batch_dist[t]);
            cudaFree(d_out_dist[t]);
            cudaFree(d_block_state[t]);
            cudaStreamDestroy(stream[t]);
        }
        cudaFree(d_src_counter);
        cudaFree(d_row_ptr);            // [優化4]
        cudaFree(d_mid);
        cudaFree(d_adj);
    };

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        perror("open output");
        release_all();
        return 1;
    }

    double gpu_and_d2h_ms = 0.0;   // [優化2] 現在代表 Host 「等 GPU(kernel + D2H) 的時間」，已與 fwrite 重疊
    double fwrite_ms = 0.0;

    const int num_batches = (n + batch_size - 1) / batch_size;

    // [優化2] 等第 b 批的 stream 完成後再 fwrite
    auto write_batch = [&](int b) {
        const int t = b & 1;
        const size_t cnt = std::min((size_t)batch_size, (size_t)n - (size_t)b * (size_t)batch_size);
        const auto t_w0 = Clock::now();
        cudaStreamSynchronize(stream[t]);
        const auto t_w1 = Clock::now();
        gpu_and_d2h_ms += Ms(t_w1 - t_w0).count();

        // [新增計時] 階段 3：fwrite 寫出結果
        if (fwrite(host_batch_dist[t], sizeof(int), cnt * (size_t)n, fout) != cnt * (size_t)n) {
            perror("write output");
            fclose(fout);
            release_all();
            exit(1);
        }
        fwrite_ms += Ms(Clock::now() - t_w1).count();
    };

    // TODO: replace whole loop with kernel launches
    // [新增 Block-per-Source] 分批啟動常駐 Block 搶算起點
    // [優化2] 不再每批 cudaDeviceSynchronize；改成 launch -> D2H(async) -> 寫上一批
    for (int b = 0; b < num_batches; ++b) {
        const int t = b & 1;
        const int src_base = b * batch_size;
        const int cur_batch_count = std::min(batch_size, n - src_base);
        const int cur_blocks = std::min(num_blocks_per, cur_batch_count);
        const size_t batch_elems = (size_t)cur_batch_count * (size_t)n;

        cudaMemsetAsync(d_src_counter + t, 0, sizeof(int), stream[t]);
        apsp_block_per_src_kernel<TPB, GRP><<<cur_blocks, TPB, 0, stream[t]>>>(
            d_row_ptr, d_mid, d_adj,        // [優化4]
            n, delta, prune,                // [優化5]
            src_base, cur_batch_count,
            d_out_dist[t], d_block_state[t], d_src_counter + t
        );

        // TODO: cudaMemcpy distances back to host
        cudaMemcpyAsync(host_batch_dist[t], d_out_dist[t], batch_elems * sizeof(int),
                        cudaMemcpyDeviceToHost, stream[t]);

        if (b > 0) write_batch(b - 1);   // 上一批的 fwrite 與這一批的 GPU 計算重疊
    }
    write_batch(num_batches - 1);
    fclose(fout);

    // TODO: cudaFree device arrays
    // ---- 釋放 Host Pinned 與 Device 記憶體 ----
    release_all();

    const double total_host_ms = Ms(Clock::now() - t_start).count();
    fprintf(stderr,
            "[Timing] read+build_CSR: %.3f ms | host_wait_gpu(sync): %.3f ms | fwrite: %.3f ms | total_host: %.3f ms\n",
            read_and_csr_ms, gpu_and_d2h_ms, fwrite_ms, total_host_ms);

    return 0;
}

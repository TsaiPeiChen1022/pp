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

// 1. 初始化距離、旗標，並將起點 src 放入初始 pending 佇列
__global__ void init_kernel(int* dist, int* flags, int* init_pend, int* d_pend_cnt, int n, int src) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    flags[v] = 0;
    if (v == src) {
        dist[v] = 0;
        flags[v] = FLAG_IN_PEND;
        int idx = atomicAdd(d_pend_cnt, 1);
        init_pend[idx] = v;
    } else {
        dist[v] = INF;
    }
}

// 2. 清除本桶子殘留的 FLAG_IN_BUCKET_DONE 旗標
__global__ void clear_bucket_done_flag_kernel(int* flags, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    atomicAnd(&flags[v], ~FLAG_IN_BUCKET_DONE);
}

// 3. 找出目前 pending 集合中、桶子編號 > current_bucket 的最小桶子編號
__global__ void find_next_bucket_kernel(const int* dist, const int* pend_verts, int pend_size,
                                       int delta, int current_bucket, int* d_min_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pend_size) return;

    int v = pend_verts[i];
    int dv = dist[v];
    if (dv >= INF) return;

    int b = dv / delta;
    if (b > current_bucket) {
        atomicMin(d_min_b, b);
    }
}

// 4. 在 GPU 上將 pending 頂點分派至：
//    - light_frontier 與 bucket_done（屬於 target_bucket）
//    - next_pend（屬於更大的 bucket，留到未來）
//    並正確更新 FLAG_IN_PEND / FLAG_IN_BUCKET_DONE
__global__ void partition_pending_kernel(const int* dist, const int* pending, int pending_size,
                                       int delta, int current_bucket, int target_bucket,
                                       int* flags,
                                       int* out_light_frontier, int* d_light_cnt,
                                       int* out_bucket_done, int* d_bucket_done_cnt,
                                       int* out_next_pend, int* d_next_pend_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pending_size) return;

    int v = pending[i];
    // 離開目前 pending 佇列，先清除 FLAG_IN_PEND
    atomicAnd(&flags[v], ~FLAG_IN_PEND);

    int dv = dist[v];
    if (dv >= INF) return;

    int b = dv / delta;
    if (b <= current_bucket) {
        return;
    } else if (b == target_bucket) {
        out_light_frontier[atomicAdd(d_light_cnt, 1)] = v;
        if ((atomicOr(&flags[v], FLAG_IN_BUCKET_DONE) & FLAG_IN_BUCKET_DONE) == 0) {
            out_bucket_done[atomicAdd(d_bucket_done_cnt, 1)] = v;
        }
    } else {
        if ((atomicOr(&flags[v], FLAG_IN_PEND) & FLAG_IN_PEND) == 0) {
            out_next_pend[atomicAdd(d_next_pend_cnt, 1)] = v;
        }
    }
}

// 5. Phase 1 每一輪結束後，直接在 GPU 將 next_light_frontier 推進為下一輪 light_frontier、
//    併入 bucket_done（去重），並清除 FLAG_IN_NEXT_LIGHT
__global__ void promote_next_light_kernel(const int* next_light, int next_light_size,
                                          int* out_light_frontier,
                                          int* bucket_done, int* d_bucket_done_cnt,
                                          int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= next_light_size) return;

    int v = next_light[i];
    out_light_frontier[i] = v;
    atomicAnd(&flags[v], ~FLAG_IN_NEXT_LIGHT);

    if ((atomicOr(&flags[v], FLAG_IN_BUCKET_DONE) & FLAG_IN_BUCKET_DONE) == 0) {
        bucket_done[atomicAdd(d_bucket_done_cnt, 1)] = v;
    }
}

// 6. [新增優化] 針對已分離的單一類別 CSR（輕邊或重邊）做鬆弛，
//    省去逐條邊判斷 w <= delta 與跳過非目標邊的分支開銷
__global__ void relax_kernel(const int* row_ptr, const Edge* adj,
                             const int* verts, int verts_size, int bound,
                             bool to_next_light,
                             int* dist, int* flags,
                             int* out_next_light, int* d_next_light_cnt,
                             int* out_pend, int* d_pend_cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= verts_size) return;

    int v = verts[idx];
    int dv = dist[v];
    if (dv >= INF) return;

    for (int e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
        int u = adj[e].to;
        int nd = dv + adj[e].w;
        if (nd < dist[u]) {
            int old = atomicMin(&dist[u], nd);
            if (nd < old) {
                if (to_next_light && nd < bound) {
                    if ((atomicOr(&flags[u], FLAG_IN_NEXT_LIGHT) & FLAG_IN_NEXT_LIGHT) == 0) {
                        int pos = atomicAdd(d_next_light_cnt, 1);
                        out_next_light[pos] = u;
                    }
                } else {
                    if ((atomicOr(&flags[u], FLAG_IN_PEND) & FLAG_IN_PEND) == 0) {
                        int pos = atomicAdd(d_pend_cnt, 1);
                        out_pend[pos] = u;
                    }
                }
            }
        }
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

    // [新增優化] 建立兩份 CSR：輕邊（w <= delta）與重邊（w > delta）分開存放
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

    const int TPB = 256;
    const int v_blocks = std::max(1, (n + TPB - 1) / TPB);

    // TODO: cudaMalloc device arrays, cudaMemcpy graph to device
    // ---- Host 端輸出緩衝區 ----
    std::vector<int> host_dist(std::max(n, 1));

    // ---- Device 端輕／重邊 CSR 與狀態陣列宣告 ----
    // [新增優化] 以輕邊與重邊兩組 CSR 取代原本單一混合的 d_row_ptr / d_adj
    int *d_light_row_ptr = nullptr, *d_heavy_row_ptr = nullptr;
    Edge *d_light_adj = nullptr, *d_heavy_adj = nullptr;
    int *d_dist = nullptr;
    int *d_flags = nullptr;

    int *d_pend_bufA = nullptr;
    int *d_pend_bufB = nullptr;
    int *d_light_frontier = nullptr;
    int *d_next_light_frontier = nullptr;
    int *d_bucket_done = nullptr;

    // d_cnts:
    // [0] = min_b
    // [1] = light_cnt
    // [2] = next_light_cnt
    // [3] = pend_cnt
    // [4] = bucket_done_cnt
    // [5] = next_pend_cnt
    int *d_cnts = nullptr;

    // ---- 配置 Device 記憶體 ----
    cudaMalloc(&d_light_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&d_heavy_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&d_light_adj, (size_t)std::max(light_m, 1) * sizeof(Edge));
    cudaMalloc(&d_heavy_adj, (size_t)std::max(heavy_m, 1) * sizeof(Edge));
    cudaMalloc(&d_dist, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_flags, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_pend_bufA, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_pend_bufB, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_light_frontier, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_next_light_frontier, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_bucket_done, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_cnts, 6 * sizeof(int));

    // ---- 將輕／重邊 CSR 資料從 Host 搬到 Device ----
    cudaMemcpy(d_light_row_ptr, light_row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heavy_row_ptr, heavy_row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (light_m > 0) {
        cudaMemcpy(d_light_adj, light_adj.data(), (size_t)light_m * sizeof(Edge), cudaMemcpyHostToDevice);
    }
    if (heavy_m > 0) {
        cudaMemcpy(d_heavy_adj, heavy_adj.data(), (size_t)heavy_m * sizeof(Edge), cudaMemcpyHostToDevice);
    }

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    // TODO: replace with flat frontier array + counter

    for (int src = 0; src < n; ++src) {
        // TODO: replace with init kernel launch
        cudaMemset(d_cnts + 3, 0, sizeof(int));
        init_kernel<<<v_blocks, TPB>>>(d_dist, d_flags, d_pend_bufA, d_cnts + 3, n, src);

        int* d_pend = d_pend_bufA;
        int* d_next_pend = d_pend_bufB;
        int pend_cnt = 1;
        int current_bucket = -1;

        // TODO: replace whole loop with kernel launches
        while (true) {
            int init_min_bucket = INT_MAX;
            cudaMemcpy(d_cnts + 0, &init_min_bucket, sizeof(int), cudaMemcpyHostToDevice);

            if (pend_cnt > 0) {
                int p_blocks = (pend_cnt + TPB - 1) / TPB;
                find_next_bucket_kernel<<<p_blocks, TPB>>>(
                    d_dist, d_pend, pend_cnt, delta, current_bucket, d_cnts + 0
                );
            }

            int next_bucket = INT_MAX;
            cudaMemcpy(&next_bucket, d_cnts + 0, sizeof(int), cudaMemcpyDeviceToHost);
            if (next_bucket == INT_MAX) break;

            clear_bucket_done_flag_kernel<<<v_blocks, TPB>>>(d_flags, n);

            int zero = 0;
            cudaMemcpy(d_cnts + 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cnts + 4, &zero, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cnts + 5, &zero, sizeof(int), cudaMemcpyHostToDevice);

            if (pend_cnt > 0) {
                int p_blocks = (pend_cnt + TPB - 1) / TPB;
                partition_pending_kernel<<<p_blocks, TPB>>>(
                    d_dist, d_pend, pend_cnt, delta, current_bucket, next_bucket,
                    d_flags,
                    d_light_frontier, d_cnts + 1,
                    d_bucket_done, d_cnts + 4,
                    d_next_pend, d_cnts + 5
                );
            }

            int light_cnt = 0;
            int bucket_done_cnt = 0;
            int next_pend_cnt = 0;
            cudaMemcpy(&light_cnt, d_cnts + 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bucket_done_cnt, d_cnts + 4, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&next_pend_cnt, d_cnts + 5, sizeof(int), cudaMemcpyDeviceToHost);

            pend_cnt = next_pend_cnt;
            cudaMemcpy(d_cnts + 3, &pend_cnt, sizeof(int), cudaMemcpyHostToDevice);

            std::swap(d_pend, d_next_pend);
            current_bucket = next_bucket;
            int bucket_upper_bound = (next_bucket + 1) * delta;

            // Phase 1: 同桶內輕邊閉包迭代（[新增優化] 直接走輕邊 CSR）
            while (light_cnt > 0) {
                cudaMemset(d_cnts + 2, 0, sizeof(int));
                int l_blocks = (light_cnt + TPB - 1) / TPB;
                relax_kernel<<<l_blocks, TPB>>>(
                    d_light_row_ptr, d_light_adj, d_light_frontier, light_cnt, bucket_upper_bound,
                    true, d_dist, d_flags,
                    d_next_light_frontier, d_cnts + 2, d_pend, d_cnts + 3
                );

                int next_light_cnt = 0;
                cudaMemcpy(&next_light_cnt, d_cnts + 2, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&pend_cnt, d_cnts + 3, sizeof(int), cudaMemcpyDeviceToHost);

                if (next_light_cnt > 0) {
                    int nl_blocks = (next_light_cnt + TPB - 1) / TPB;
                    promote_next_light_kernel<<<nl_blocks, TPB>>>(
                        d_next_light_frontier, next_light_cnt,
                        d_light_frontier, d_bucket_done, d_cnts + 4, d_flags
                    );
                    cudaMemcpy(&bucket_done_cnt, d_cnts + 4, sizeof(int), cudaMemcpyDeviceToHost);
                }
                light_cnt = next_light_cnt;
            }

            // Phase 2: 對本桶子累積的所有頂點做重邊鬆弛（[新增優化] 直接走重邊 CSR）
            if (bucket_done_cnt > 0 && heavy_m > 0) {
                int bd_blocks = (bucket_done_cnt + TPB - 1) / TPB;
                relax_kernel<<<bd_blocks, TPB>>>(
                    d_heavy_row_ptr, d_heavy_adj, d_bucket_done, bucket_done_cnt, bucket_upper_bound,
                    false, d_dist, d_flags,
                    d_next_light_frontier, d_cnts + 2, d_pend, d_cnts + 3
                );
                cudaMemcpy(&pend_cnt, d_cnts + 3, sizeof(int), cudaMemcpyDeviceToHost);
            }
        }

        cudaDeviceSynchronize();

        // TODO: cudaMemcpy distances back to host
        cudaMemcpy(host_dist.data(), d_dist, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);

        if (fwrite(host_dist.data(), sizeof(int), n, fout) != (size_t)n) {
            perror("write output");
            return 1;
        }
    }
    fclose(fout);

    // TODO: cudaFree device arrays
    // ---- 釋放 Device 記憶體 ----
    cudaFree(d_light_row_ptr);
    cudaFree(d_heavy_row_ptr);
    cudaFree(d_light_adj);
    cudaFree(d_heavy_adj);
    cudaFree(d_dist);
    cudaFree(d_flags);
    cudaFree(d_pend_bufA);
    cudaFree(d_pend_bufB);
    cudaFree(d_light_frontier);
    cudaFree(d_next_light_frontier);
    cudaFree(d_bucket_done);
    cudaFree(d_cnts);

    return 0;
}
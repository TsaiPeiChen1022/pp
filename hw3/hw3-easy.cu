#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

const int INF = (1 << 30) - 1;

struct Edge {
    int to;
    int w;
};

// TODO: add kernels: init distances, relax frontier edges (atomicMin / atomicAdd), find next bucket

// 1. 初始化距離、旗標，並將起點 src 放入初始 frontier
__global__ void init_kernel(int* dist, int* flags, int* frontier, int* d_cnt, int n, int src) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    flags[v] = 0;
    if (v == src) {
        dist[v] = 0;
        flags[v] = 1; // 標記已在 pending/frontier 中
        int idx = atomicAdd(d_cnt, 1);
        frontier[idx] = v;
    } else {
        dist[v] = INF;
    }
}

// 2. 找出目前候選集合中、桶子編號 > current_bucket 的最小桶子編號
__global__ void find_next_bucket_kernel(const int* dist, const int* verts, int verts_size,
                                       int delta, int current_bucket, int* d_min_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= verts_size) return;

    int v = verts[i];
    int dv = dist[v];
    if (dv >= INF) return;

    int b = dv / delta;
    if (b > current_bucket) {
        atomicMin(d_min_b, b);
    }
}

// 3. 鬆弛邊（輕邊或重邊），並以 atomicMin 更新距離、atomicAdd 收集下一批頂點
__global__ void relax_kernel(const int* row_ptr, const Edge* adj, int delta,
                             const int* verts, int verts_size, int bound,
                             bool is_light, bool to_cur,
                             int* dist, int* flags,
                             int* out_cur, int* d_cur_cnt,
                             int* out_pend, int* d_pend_cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= verts_size) return;

    int v = verts[idx];
    int dv = dist[v];
    if (dv >= INF) return;

    for (int e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
        int w = adj[e].w;
        bool edge_light = (w <= delta);
        if (edge_light != is_light) continue;

        int u = adj[e].to;
        int nd = dv + w;
        if (nd < dist[u]) {
            int old = atomicMin(&dist[u], nd);
            if (nd < old) {
                // 輕邊且新距離仍在目前 bucket 範圍內，且要求進 cur（Phase 1 迭代）
                if (is_light && to_cur && nd < bound) {
                    // bit 1 (2) 代表已在下一批 cur 中
                    if ((atomicOr(&flags[u], 2) & 2) == 0) {
                        int pos = atomicAdd(d_cur_cnt, 1);
                        out_cur[pos] = u;
                    }
                } else {
                    // 否則放入 pending 等待後續 bucket 處理（bit 0 (1) 代表已在 pending 中）
                    if ((atomicOr(&flags[u], 1) & 1) == 0) {
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

    // Build CSR: out-edges of v are adj[row_ptr[v] .. row_ptr[v+1])
    std::vector<int> row_ptr(n + 1, 0);
    for (int i = 0; i < m; ++i) row_ptr[raw[3 * i] + 1]++;        // out-degree counts
    for (int v = 0; v < n; ++v) row_ptr[v + 1] += row_ptr[v];     // prefix sum

    std::vector<Edge> adj(std::max(m, 1));
    std::vector<int> next(row_ptr.begin(), row_ptr.end() - 1);    // next free slot per vertex
    for (int i = 0; i < m; ++i) {
        const int src = raw[3 * i];
        adj[next[src]++] = Edge{raw[3 * i + 1], raw[3 * i + 2]};
    }

    // Bucket width: same default as hw3-1.cu
    long long wsum = 0;
    for (int i = 0; i < m; ++i) wsum += raw[3 * i + 2];
    const double avg_w = m > 0 ? (double)wsum / m : 1.0;
    const double avg_deg = n > 0 ? std::max(1.0, (double)m / n) : 1.0;
    int delta = std::max(1, (int)(avg_w * 32.0 / avg_deg));
    if (const char* s = getenv("HW3_DELTA")) delta = std::max(1, atoi(s));

    const int TPB = 256;
    const int v_blocks = std::max(1, (n + TPB - 1) / TPB);

    // TODO: cudaMalloc device arrays, cudaMemcpy graph to device
    // ---- Host 端輔助陣列 ----
    std::vector<int> d(n);          // 存放單一來源算出的最短距離結果，準備寫入輸出檔
    std::vector<int> h_pend(n);     // Host 端暫存 pending 佇列，用來判斷與分派頂點

    // ---- Device 端圖結構與狀態陣列宣告 ----
    int *d_row_ptr = nullptr;       // CSR 列的起始索引陣列，大小 (n + 1) * sizeof(int)
    Edge *d_adj = nullptr;          // CSR 鄰接邊陣列，大小 max(m, 1) * sizeof(Edge)
    int *d_dist = nullptr;          // 目前起點到各頂點的最短距離，大小 max(n, 1) * sizeof(int)
    int *d_flags = nullptr;         // 各頂點狀態旗標（去重 / 記錄是否已在佇列或 R 集合），大小 max(n, 1) * sizeof(int)

    int *d_pendA = nullptr;         // Pending 雙緩衝區 A（存放尚未處理或未來 bucket 的頂點），大小 max(n, 1) * sizeof(int)
    int *d_pendB = nullptr;         // Pending 雙緩衝區 B，與 d_pendA 交替交換使用，大小 max(n, 1) * sizeof(int)
    int *d_cur = nullptr;           // 當前輕邊閉包迭代要處理的頂點集合，大小 max(n, 1) * sizeof(int)
    int *d_nxt = nullptr;           // 輕邊鬆弛後、仍落在同一個 bucket 內下一輪要處理的頂點，大小 max(n, 1) * sizeof(int)
    int *d_R = nullptr;             // 本 bucket 內所有處理過的頂點集合（Phase 2 重邊鬆弛會用到），大小 max(n, 1) * sizeof(int)

    // 原子計數器陣列（共 6 個 int）：
    // [0] = min_b       : 下一輪要處理的最小 bucket 編號
    // [1] = cur_cnt     : d_cur 中的頂點數量
    // [2] = nxt_cnt     : d_nxt 中的頂點數量
    // [3] = pend_cnt    : pending 佇列中的頂點數量
    // [4] = R_cnt       : d_R 中的頂點數量
    // [5] = (保留/備用) : 可視需要擴充計數器
    int *d_cnts = nullptr;

    // ---- 配置 Device 記憶體 ----
    cudaMalloc(&d_row_ptr, (size_t)(n + 1) * sizeof(int));
    cudaMalloc(&d_adj, (size_t)std::max(m, 1) * sizeof(Edge));
    cudaMalloc(&d_dist, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_flags, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_pendA, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_pendB, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_cur, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_nxt, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_R, (size_t)std::max(n, 1) * sizeof(int));
    cudaMalloc(&d_cnts, 6 * sizeof(int));

    // ---- 將圖資料從 Host 搬到 Device ----
    cudaMemcpy(d_row_ptr, row_ptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (m > 0) {
        cudaMemcpy(d_adj, adj.data(), (size_t)m * sizeof(Edge), cudaMemcpyHostToDevice);
    }

    // Delta-stepping from every source
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    // TODO: replace with flat frontier array + counter

    for (int src = 0; src < n; ++src) {
        // TODO: replace with init kernel launch
        cudaMemset(d_cnts + 3, 0, sizeof(int));
        init_kernel<<<v_blocks, TPB>>>(d_dist, d_flags, d_pendA, d_cnts + 3, n, src);

        int* d_pend = d_pendA;
        int* d_pnew = d_pendB;
        int pend_cnt = 1;
        int cb = -1;

        // TODO: replace whole loop with kernel launches
        while (true) {
            int h_min_init = INT_MAX;
            cudaMemcpy(d_cnts + 0, &h_min_init, sizeof(int), cudaMemcpyHostToDevice);

            if (pend_cnt > 0) {
                int p_blocks = (pend_cnt + TPB - 1) / TPB;
                find_next_bucket_kernel<<<p_blocks, TPB>>>(d_dist, d_pend, pend_cnt, delta, cb, d_cnts + 0);
            }

            int minb = INT_MAX;
            cudaMemcpy(&minb, d_cnts + 0, sizeof(int), cudaMemcpyDeviceToHost);
            if (minb == INT_MAX) break;

            // 依目前距離把屬於 minb 的頂點挑進 h_cur 與 h_R，其餘保留到 h_next_pend
            cudaMemcpy(h_pend.data(), d_pend, (size_t)pend_cnt * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(d.data(), d_dist, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);

            std::vector<int> h_cur, h_R, h_next_pend;
            for (int i = 0; i < pend_cnt; ++i) {
                int v = h_pend[i];
                int dv = d[v];
                if (dv >= INF) continue;
                int b = dv / delta;
                if (b <= cb) {
                    // 過期頂點，清除旗標
                    cudaMemset(d_flags + v, 0, sizeof(int));
                } else if (b == minb) {
                    h_cur.push_back(v);
                    h_R.push_back(v);
                } else {
                    h_next_pend.push_back(v);
                }
            }

            int cur_cnt = (int)h_cur.size();
            int R_cnt = (int)h_R.size();
            pend_cnt = (int)h_next_pend.size();

            if (cur_cnt > 0) {
                cudaMemcpy(d_cur, h_cur.data(), (size_t)cur_cnt * sizeof(int), cudaMemcpyHostToDevice);
            }
            if (R_cnt > 0) {
                cudaMemcpy(d_R, h_R.data(), (size_t)R_cnt * sizeof(int), cudaMemcpyHostToDevice);
            }
            if (pend_cnt > 0) {
                cudaMemcpy(d_pnew, h_next_pend.data(), (size_t)pend_cnt * sizeof(int), cudaMemcpyHostToDevice);
            }
            cudaMemcpy(d_cnts + 3, &pend_cnt, sizeof(int), cudaMemcpyHostToDevice);

            std::swap(d_pend, d_pnew);
            cb = minb;
            int bound = (minb + 1) * delta;

            // Phase 1: light-edge closure
            while (cur_cnt > 0) {
                cudaMemset(d_cnts + 2, 0, sizeof(int)); // nxt_cnt = 0
                int c_blocks = (cur_cnt + TPB - 1) / TPB;
                relax_kernel<<<c_blocks, TPB>>>(
                    d_row_ptr, d_adj, delta, d_cur, cur_cnt, bound,
                    true, true, d_dist, d_flags, d_nxt, d_cnts + 2, d_pend, d_cnts + 3
                );

                int nxt_cnt = 0;
                cudaMemcpy(&nxt_cnt, d_cnts + 2, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&pend_cnt, d_cnts + 3, sizeof(int), cudaMemcpyDeviceToHost);

                if (nxt_cnt > 0) {
                    // 將 nxt 併入 R，並清除 nxt 的 F_NEXT (bit 1) 旗標，準備下一輪輕邊迭代
                    std::vector<int> h_nxt(nxt_cnt);
                    cudaMemcpy(h_nxt.data(), d_nxt, (size_t)nxt_cnt * sizeof(int), cudaMemcpyDeviceToHost);
                    for (int v : h_nxt) {
                        h_R.push_back(v);
                        int f = 0;
                        cudaMemcpy(&f, d_flags + v, sizeof(int), cudaMemcpyDeviceToHost);
                        f &= ~2;
                        cudaMemcpy(d_flags + v, &f, sizeof(int), cudaMemcpyHostToDevice);
                    }
                    R_cnt = (int)h_R.size();
                    cudaMemcpy(d_R, h_R.data(), (size_t)R_cnt * sizeof(int), cudaMemcpyHostToDevice);
                    std::swap(d_cur, d_nxt);
                }
                cur_cnt = nxt_cnt;
            }

            // Phase 2: heavy-edge relaxation
            if (R_cnt > 0) {
                int r_blocks = (R_cnt + TPB - 1) / TPB;
                relax_kernel<<<r_blocks, TPB>>>(
                    d_row_ptr, d_adj, delta, d_R, R_cnt, bound,
                    false, false, d_dist, d_flags, d_nxt, d_cnts + 2, d_pend, d_cnts + 3
                );
                cudaMemcpy(&pend_cnt, d_cnts + 3, sizeof(int), cudaMemcpyDeviceToHost);
            }
        }

        cudaDeviceSynchronize();

        // TODO: cudaMemcpy distances back to host
        cudaMemcpy(d.data(), d_dist, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);

        // Row src of the answer; unreachable vertices keep INF.
        if (fwrite(d.data(), sizeof(int), n, fout) != (size_t)n) {
            perror("write output");
            return 1;
        }
    }
    fclose(fout);

    // TODO: cudaFree device arrays
    // ---- 釋放 Device 記憶體 ----
    cudaFree(d_row_ptr);
    cudaFree(d_adj);
    cudaFree(d_dist);
    cudaFree(d_flags);
    cudaFree(d_pendA);
    cudaFree(d_pendB);
    cudaFree(d_cur);
    cudaFree(d_nxt);
    cudaFree(d_R);
    cudaFree(d_cnts);

    return 0;
}
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda.h>

const int INF = (1 << 30) - 1;

struct Edge {
    int to;
    int w;
};

// CUDA Kernel: 初始化每個來源點的距離矩陣
__global__ void init_distances_kernel(int* d_dist, int n, int src) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        d_dist[idx] = (idx == src) ? 0 : INF;
    }
}

// CUDA Kernel: 輕邊鬆弛（Light-edge relaxation with Delta-stepping）
__global__ void light_relaxation_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ edge_to,
    const int* __restrict__ edge_w,
    int* __restrict__ d_dist,
    const int* __restrict__ current_frontier,
    int frontier_size,
    int delta,
    int current_bucket_idx,
    int* __restrict__ next_bucket_flags,
    int* __restrict__ updated_flag
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = current_frontier[tid];
    int u_dist = d_dist[u];

    // 檢查是否為過期（Stale）頂點
    if (u_dist / delta != current_bucket_idx) return;

    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int e = start; e < end; ++e) {
        int v = edge_to[e];
        int w = edge_w[e];

        // 僅處理輕邊 (w <= delta)
        if (w <= delta) {
            int new_dist = u_dist + w;
            int old_dist = atomicMin(&d_dist[v], new_dist);
            if (new_dist < old_dist) {
                int b = new_dist / delta;
                next_bucket_flags[v] = b;
                *updated_flag = 1;
            }
        }
    }
}

// CUDA Kernel: 重邊鬆弛（Heavy-edge relaxation）
__global__ void heavy_relaxation_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ edge_to,
    const int* __restrict__ edge_w,
    int* __restrict__ d_dist,
    const int* __restrict__ bucket_nodes,
    int num_nodes,
    int delta,
    int current_bucket_idx,
    int* __restrict__ next_bucket_flags,
    int* __restrict__ updated_flag
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int u = bucket_nodes[tid];
    int u_dist = d_dist[u];

    if (u_dist / delta != current_bucket_idx) return;

    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int e = start; e < end; ++e) {
        int v = edge_to[e];
        int w = edge_w[e];

        // 僅處理重邊 (w > delta)
        if (w > delta) {
            int new_dist = u_dist + w;
            int old_dist = atomicMin(&d_dist[v], new_dist);
            if (new_dist < old_dist) {
                int b = new_dist / delta;
                next_bucket_flags[v] = b;
                *updated_flag = 1;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    // 讀取輸入圖: n, m, 以及 m 條邊 (src, dst, weight)
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 1; }
    int n, m;
    if (fread(&n, sizeof(int), 1, fin) != 1 || fread(&m, sizeof(int), 1, fin) != 1) {
        fprintf(stderr, "bad input header\n");
        fclose(fin);
        return 1;
    }
    std::vector<int> raw((size_t)m * 3);
    if (fread(raw.data(), sizeof(int), raw.size(), fin) != raw.size()) {
        fprintf(stderr, "truncated input\n");
        fclose(fin);
        return 1;
    }
    fclose(fin);

    // 建立 CSR 格式
    std::vector<int> row_ptr(n + 1, 0);
    for (int i = 0; i < m; ++i) row_ptr[raw[3 * i] + 1]++;
    for (int v = 0; v < n; ++v) row_ptr[v + 1] += row_ptr[v];

    std::vector<int> edge_to(m);
    std::vector<int> edge_w(m);
    std::vector<int> next(row_ptr.begin(), row_ptr.end() - 1);
    for (int i = 0; i < m; ++i) {
        const int src = raw[3 * i];
        int idx = next[src]++;
        edge_to[idx] = raw[3 * i + 1];
        edge_w[idx] = raw[3 * i + 2];
    }

    // 計算 Delta 參數
    long long wsum = 0;
    for (int i = 0; i < m; ++i) wsum += raw[3 * i + 2];
    const double avg_w = m > 0 ? (double)wsum / m : 1.0;
    const double avg_deg = n > 0 ? std::max(1.0, (double)m / n) : 1.0;
    int delta = std::max(1, (int)(avg_w * 32.0 / avg_deg));
    if (const char* s = getenv("HW3_DELTA")) delta = std::max(1, atoi(s));

    // 配置 Device 記憶體
    int *d_row_ptr, *d_edge_to, *d_edge_w, *d_dist;
    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_edge_to, m * sizeof(int));
    cudaMalloc(&d_edge_w, m * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_to, edge_to.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_w, edge_w.data(), m * sizeof(int), cudaMemcpyHostToDevice);

    // 輔助 Device 陣列
    int *d_frontier, *d_next_bucket_flags, *d_updated_flag;
    cudaMalloc(&d_frontier, n * sizeof(int));
    cudaMalloc(&d_next_bucket_flags, n * sizeof(int));
    cudaMalloc(&d_updated_flag, sizeof(int));

    std::vector<int> h_dist(n);
    std::vector<int> h_next_flags(n);

    // 開啟輸出檔案
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    int threadsPerBlock = 256;

    // 對每一個源點執行 Delta-stepping 尋找單源最短路徑
    for (int src = 0; src < n; ++src) {
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        init_distances_kernel<<<blocks, threadsPerBlock>>>(d_dist, n, src);
        cudaDeviceSynchronize();

        int max_bucket = n * 1000 / delta + 1;

        for (int i = 0; i < max_bucket; ++i) {
            // 收集目前區桶 i 中的頂點
            cudaMemcpy(h_dist.data(), d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
            std::vector<int> current_bucket_nodes;
            for (int v = 0; v < n; ++v) {
                if (h_dist[v] / delta == i) {
                    current_bucket_nodes.push_back(v);
                }
            }

            if (current_bucket_nodes.empty()) {
                // 檢查是否還有更高索引的非空區桶，若無則結束
                bool has_more = false;
                for (int v = 0; v < n; ++v) {
                    if (h_dist[v] != INF && h_dist[v] / delta > i) {
                        has_more = true;
                        break;
                    }
                }
                if (!has_more) break;
                continue;
            }

            // Phase 1: 輕邊封閉 (Light-edge closure)
            while (true) {
                cudaMemcpy(h_dist.data(), d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
                current_bucket_nodes.clear();
                for (int v = 0; v < n; ++v) {
                    if (h_dist[v] / delta == i) {
                        current_bucket_nodes.push_back(v);
                    }
                }
                if (current_bucket_nodes.empty()) break;

                cudaMemcpy(d_frontier, current_bucket_nodes.data(), current_bucket_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);
                
                int f_blocks = (current_bucket_nodes.size() + threadsPerBlock - 1) / threadsPerBlock;
                int h_updated = 0;
                cudaMemcpy(d_updated_flag, &h_updated, sizeof(int), cudaMemcpyHostToDevice);

                light_relaxation_kernel<<<f_blocks, threadsPerBlock>>>(
                    d_row_ptr, d_edge_to, d_edge_w, d_dist,
                    d_frontier, current_bucket_nodes.size(), delta, i,
                    d_next_bucket_flags, d_updated_flag
                );
                cudaDeviceSynchronize();

                cudaMemcpy(&h_updated, d_updated_flag, sizeof(int), cudaMemcpyDeviceToHost);
                if (!h_updated) break;
            }

            // Phase 2: 重邊鬆弛 (Heavy-edge relaxation)
            cudaMemcpy(h_dist.data(), d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
            std::vector<int> r_i_nodes;
            for (int v = 0; v < n; ++v) {
                if (h_dist[v] / delta == i) {
                    r_i_nodes.push_back(v);
                }
            }

            if (!r_i_nodes.empty()) {
                cudaMemcpy(d_frontier, r_i_nodes.data(), r_i_nodes.size() * sizeof(int), cudaMemcpyHostToDevice);
                int r_blocks = (r_i_nodes.size() + threadsPerBlock - 1) / threadsPerBlock;

                heavy_relaxation_kernel<<<r_blocks, threadsPerBlock>>>(
                    d_row_ptr, d_edge_to, d_edge_w, d_dist,
                    d_frontier, r_i_nodes.size(), delta, i,
                    d_next_bucket_flags, d_updated_flag
                );
                cudaDeviceSynchronize();
            }
        }

        // 將結果寫回主機並輸出到檔案
        cudaMemcpy(h_dist.data(), d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (fwrite(h_dist.data(), sizeof(int), n, fout) != (size_t)n) {
            perror("write output");
            fclose(fout);
            return 1;
        }
    }

    fclose(fout);

    // 釋放 Device 記憶體
    cudaFree(d_row_ptr);
    cudaFree(d_edge_to);
    cudaFree(d_edge_w);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_next_bucket_flags);
    cudaFree(d_updated_flag);

    return 0;
}
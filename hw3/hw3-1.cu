#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

const int INF = (1 << 30) - 1;
#define THREADS 256
#define F_PEND 1   // in pending list
#define F_NEXT 2   // in next light queue
#define F_R    4   // in R (settled this round)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

__device__ __forceinline__ int warp_min(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// One block = one source, whole delta-stepping runs inside the block.
// out row (sidx) is used directly as dist[].
// state per block: flags, pendA, pendB, cur, nxt, R  (each n ints)
template <int G>
__global__ void __launch_bounds__(THREADS)
sssp_kernel(const int* __restrict__ row_ptr, const int* __restrict__ mid,
            const int2* __restrict__ adj, int n, int delta,
            int src_base, int src_count,
            int* __restrict__ out, int* __restrict__ state, int* __restrict__ counter)
{
    __shared__ int s_src, s_min, s_R, s_cur;
    __shared__ int s_pc[2], s_nxt[2];

    const int tid = threadIdx.x;
    int* base  = state + (size_t)blockIdx.x * 6 * n;
    int* flags = base;
    int* pendA = base + (size_t)n;
    int* pendB = base + (size_t)2 * n;
    int* cur   = base + (size_t)3 * n;
    int* nxt   = base + (size_t)4 * n;
    int* R     = base + (size_t)5 * n;

    const int grp = tid / G, lane = tid % G, ngrp = THREADS / G;

    while (true) {
        if (tid == 0) s_src = atomicAdd(counter, 1);
        __syncthreads();
        const int sidx = s_src;
        if (sidx >= src_count) break;

        int* dist = out + (size_t)sidx * n;
        const int s = src_base + sidx;

        for (int v = tid; v < n; v += THREADS) dist[v] = INF;
        __syncthreads();

        int* pend = pendA;
        int* pnew = pendB;
        if (tid == 0) {
            dist[s] = 0;
            pend[0] = s;
            flags[s] = F_PEND;
            s_pc[0] = 1; s_pc[1] = 0;
            s_nxt[0] = 0; s_nxt[1] = 0;
        }
        __syncthreads();

        int rp = 0, k = 0, cb = -1;

        while (true) {
            // ---------- find lowest non-empty bucket ----------
            if (tid == 0) { s_min = INT_MAX; s_R = 0; s_cur = 0; s_pc[rp ^ 1] = 0; }
            const int pc = s_pc[rp];
            __syncthreads();

            int lm = INT_MAX;
            for (int i = tid; i < pc; i += THREADS) {
                int b = __ldcg(&dist[pend[i]]) / delta;
                if (b > cb) lm = min(lm, b);
            }
            lm = warp_min(lm);
            if ((tid & 31) == 0 && lm != INT_MAX) atomicMin(&s_min, lm);
            __syncthreads();
            const int minb = s_min;

            // ---------- partition pending: drop stale / extract bucket minb / keep ----------
            for (int i = tid; i < pc; i += THREADS) {
                int v = pend[i];
                int b = __ldcg(&dist[v]) / delta;
                if (b <= cb) {
                    flags[v] = 0;                               // stale entry
                } else if (b == minb) {
                    flags[v] = F_R;
                    cur[atomicAdd(&s_cur, 1)] = v;
                    R[atomicAdd(&s_R, 1)] = v;
                } else {
                    pnew[atomicAdd(&s_pc[rp ^ 1], 1)] = v;
                }
            }
            __syncthreads();
            if (minb == INT_MAX) break;                         // this source is done

            { int* t = pend; pend = pnew; pnew = t; }
            rp ^= 1;
            cb = minb;
            int cc = s_cur;
            const int bound = (minb + 1) * delta;               // nd < bound  <=> bucket == minb

            // ---------- Phase 1: light-edge closure ----------
            while (cc > 0) {
                const int par = k & 1;
                for (int i = grp; i < cc; i += ngrp) {
                    int v = cur[i];
                    int dv = __ldcg(&dist[v]);
                    int e_end = mid[v];
                    for (int e = row_ptr[v] + lane; e < e_end; e += G) {
                        int2 ed = __ldg(&adj[e]);
                        int nd = dv + ed.y;
                        if (nd >= __ldcg(&dist[ed.x])) continue;
                        int old = atomicMin(&dist[ed.x], nd);
                        if (nd < old) {
                            if (nd < bound) {
                                if (!(atomicOr(&flags[ed.x], F_NEXT) & F_NEXT))
                                    nxt[atomicAdd(&s_nxt[par], 1)] = ed.x;
                            } else {
                                if (!(atomicOr(&flags[ed.x], F_PEND) & F_PEND))
                                    pend[atomicAdd(&s_pc[rp], 1)] = ed.x;
                            }
                        }
                    }
                }
                __syncthreads();
                const int nn = s_nxt[par];
                if (tid == 0) s_nxt[par ^ 1] = 0;
                for (int i = tid; i < nn; i += THREADS) {
                    int v = nxt[i];
                    cur[i] = v;
                    int f = flags[v];
                    if (!(f & F_R)) R[atomicAdd(&s_R, 1)] = v;
                    flags[v] = (f | F_R) & ~F_NEXT;
                }
                __syncthreads();
                cc = nn;
                ++k;
            }

            // ---------- Phase 2: heavy edges of everything settled in R ----------
            const int rc = s_R;
            for (int i = grp; i < rc; i += ngrp) {
                int v = R[i];
                int dv = __ldcg(&dist[v]);
                int e_end = row_ptr[v + 1];
                for (int e = mid[v] + lane; e < e_end; e += G) {
                    int2 ed = __ldg(&adj[e]);
                    int nd = dv + ed.y;
                    if (nd >= __ldcg(&dist[ed.x])) continue;
                    int old = atomicMin(&dist[ed.x], nd);
                    if (nd < old) {
                        if (!(atomicOr(&flags[ed.x], F_PEND) & F_PEND))
                            pend[atomicAdd(&s_pc[rp], 1)] = ed.x;
                    }
                }
            }
            __syncthreads();
            for (int i = tid; i < rc; i += THREADS) flags[R[i]] &= ~F_R;
            __syncthreads();
        }
        __syncthreads();
    }
}

typedef void (*KFn)(const int*, const int*, const int2*, int, int, int, int, int*, int*, int*);

int main(int argc, char* argv[]) {
    if (argc < 3) { fprintf(stderr, "usage: %s <input> <output>\n", argv[0]); return 1; }

    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 1; }
    int n, m;
    if (fread(&n, sizeof(int), 1, fin) != 1 || fread(&m, sizeof(int), 1, fin) != 1) {
        fprintf(stderr, "bad input header\n"); return 1;
    }
    std::vector<int> raw((size_t)m * 3);
    if (fread(raw.data(), sizeof(int), raw.size(), fin) != raw.size()) {
        fprintf(stderr, "truncated input\n"); return 1;
    }
    fclose(fin);

    // delta (same default as the template)
    long long wsum = 0;
    for (int i = 0; i < m; ++i) wsum += raw[3 * i + 2];
    const double avg_w = m > 0 ? (double)wsum / m : 1.0;
    const double avg_deg = n > 0 ? std::max(1.0, (double)m / n) : 1.0;
    int delta = std::max(1, (int)(avg_w * 32.0 / avg_deg));
    if (const char* s = getenv("HW3_DELTA")) delta = std::max(1, atoi(s));

    // CSR, light edges first within each vertex
    std::vector<int> row_ptr(n + 1, 0), mid(n);
    for (int i = 0; i < m; ++i) row_ptr[raw[3 * i] + 1]++;
    for (int v = 0; v < n; ++v) row_ptr[v + 1] += row_ptr[v];
    std::vector<int2> adj(m);
    {
        std::vector<int> nx(row_ptr.begin(), row_ptr.end() - 1);
        for (int i = 0; i < m; ++i) {
            int idx = nx[raw[3 * i]]++;
            adj[idx] = make_int2(raw[3 * i + 1], raw[3 * i + 2]);
        }
    }
    std::vector<int>().swap(raw);
    for (int v = 0; v < n; ++v) {
        int2* b = adj.data() + row_ptr[v];
        int2* e = adj.data() + row_ptr[v + 1];
        int2* p = std::stable_partition(b, e, [delta](const int2& x) { return x.y <= delta; });
        mid[v] = row_ptr[v] + (int)(p - b);
    }

    // device graph
    int *d_row, *d_mid; int2* d_adj;
    CK(cudaMalloc(&d_row, (n + 1) * sizeof(int)));
    CK(cudaMalloc(&d_mid, n * sizeof(int)));
    CK(cudaMalloc(&d_adj, (size_t)std::max(m, 1) * sizeof(int2)));
    CK(cudaMemcpy(d_row, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_mid, mid.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    if (m > 0) CK(cudaMemcpy(d_adj, adj.data(), (size_t)m * sizeof(int2), cudaMemcpyHostToDevice));
    std::vector<int2>().swap(adj);

    // pick sub-warp size by average degree
    KFn kern;
    if (avg_deg < 6)       kern = sssp_kernel<4>;
    else if (avg_deg < 16) kern = sssp_kernel<8>;
    else if (avg_deg < 48) kern = sssp_kernel<16>;
    else                   kern = sssp_kernel<32>;

    int dev = 0, numSM = 0, occ = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, THREADS, 0));
    if (occ < 1) occ = 1;

    // two streams, each with its own block state and output buffer
    size_t stateBytesPerBlock = (size_t)6 * n * sizeof(int);
    int gridPer = std::max(1, numSM * occ / 2);
    while (gridPer > 1 && (size_t)gridPer * 2 * stateBytesPerBlock > ((size_t)3 << 30)) gridPer /= 2;

    size_t batch = ((size_t)400 << 20) / ((size_t)n * sizeof(int));
    batch = std::max<size_t>(batch, (size_t)gridPer * 2);
    batch = std::min<size_t>(batch, (size_t)n);

    int* d_state[2]; int* d_out[2]; int* h_out[2]; int* d_cnt;
    cudaStream_t st[2];
    CK(cudaMalloc(&d_cnt, 2 * sizeof(int)));
    for (int t = 0; t < 2; ++t) {
        CK(cudaStreamCreate(&st[t]));
        CK(cudaMalloc(&d_state[t], (size_t)gridPer * stateBytesPerBlock));
        CK(cudaMemset(d_state[t], 0, (size_t)gridPer * stateBytesPerBlock));   // flags must start at 0
        CK(cudaMalloc(&d_out[t], batch * n * sizeof(int)));
        CK(cudaMallocHost(&h_out[t], batch * n * sizeof(int)));
    }

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    int numBatches = (int)(((size_t)n + batch - 1) / batch);
    auto write_batch = [&](int b) {
        int t = b & 1;
        CK(cudaStreamSynchronize(st[t]));
        size_t cnt = std::min<size_t>(batch, (size_t)n - (size_t)b * batch);
        if (fwrite(h_out[t], sizeof(int), cnt * (size_t)n, fout) != cnt * (size_t)n) {
            perror("write output"); exit(1);
        }
    };

    for (int b = 0; b < numBatches; ++b) {
        int t = b & 1;
        int base = (int)((size_t)b * batch);
        int cnt = (int)std::min<size_t>(batch, (size_t)n - base);
        int grid = std::min(gridPer, cnt);
        CK(cudaMemsetAsync(d_cnt + t, 0, sizeof(int), st[t]));
        kern<<<grid, THREADS, 0, st[t]>>>(d_row, d_mid, d_adj, n, delta, base, cnt,
                                         d_out[t], d_state[t], d_cnt + t);
        CK(cudaMemcpyAsync(h_out[t], d_out[t], (size_t)cnt * n * sizeof(int),
                           cudaMemcpyDeviceToHost, st[t]));
        if (b > 0) write_batch(b - 1);      // overlap file write with GPU work
    }
    write_batch(numBatches - 1);
    CK(cudaGetLastError());

    fclose(fout);
    for (int t = 0; t < 2; ++t) {
        cudaFree(d_state[t]); cudaFree(d_out[t]); cudaFreeHost(h_out[t]); cudaStreamDestroy(st[t]);
    }
    cudaFree(d_row); cudaFree(d_mid); cudaFree(d_adj); cudaFree(d_cnt);
    return 0;
}
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

// TODO: add kernels: init distances, relax frontier edges (atomicMin / atomicAdd), find next bucket

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    // TODO: get device properties

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

    std::vector<Edge> adj(m);
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

    // TODO: cudaMalloc device arrays, cudaMemcpy graph to device

    // Delta-stepping from every source
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    std::vector<int> d(n);
    std::vector<std::vector<int>> buckets;   // TODO: replace with flat frontier array + counter

    for (int src = 0; src < n; ++src) {
        // TODO: replace with init kernel launch
        std::fill(d.begin(), d.end(), INF);
        d[src] = 0;
        buckets.clear();
        buckets.push_back({src});                       // bucket 0 holds the source

        // TODO: replace whole loop with kernel launches
        for (size_t i = 0; i < buckets.size(); ++i) {
            while (!buckets[i].empty()) {
                std::vector<int> current;
                current.swap(buckets[i]);               // take everything out of bucket i

                for (int v : current) {
                    if (d[v] / delta != (int)i) continue;   // stale copy, v moved to a lower bucket

                    for (int e = row_ptr[v]; e < row_ptr[v + 1]; ++e) {
                        const int u = adj[e].to;
                        const int nd = d[v] + adj[e].w;
                        if (nd < d[u]) {
                            d[u] = nd;
                            const size_t b = nd / delta;
                            if (b >= buckets.size()) buckets.resize(b + 1);
                            buckets[b].push_back(u);
                        }
                    }
                }
            }
        }

        // TODO: cudaMemcpy distances back to host

        // Row src of the answer; unreachable vertices keep INF.
        if (fwrite(d.data(), sizeof(int), n, fout) != (size_t)n) {
            perror("write output");
            return 1;
        }
    }
    fclose(fout);

    // TODO: cudaFree device arrays

    return 0;
}
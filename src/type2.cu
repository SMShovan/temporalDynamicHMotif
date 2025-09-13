#include <iostream>
#include <vector>
#include <climits>
#include <cuda_runtime.h>

#include "../include/graphGeneration.hpp"
#include "../include/utils.hpp"
#include "../include/structure.hpp"

// Local flattener with padding-to-4 and INT_MIN sentinel as used in this project
static std::pair<std::vector<int>, std::vector<int>> localFlatten2D(const std::vector<std::vector<int>>& vec2d) {
    auto nextMultipleOf4 = [](int x) { return ((x + 3) / 4) * 4; };
    std::vector<int> flatValues;
    std::vector<int> startOffsets(vec2d.size());
    int cursor = 0;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        startOffsets[i] = cursor;
        int inner = static_cast<int>(vec2d[i].size());
        int padded = nextMultipleOf4(inner);
        for (int j = 0; j < padded; ++j) {
            if (j < inner) flatValues.push_back(vec2d[i][j]);
            else if (j == padded - 1) flatValues.push_back(INT_MIN);
            else flatValues.push_back(0);
            ++cursor;
        }
    }
    return {flatValues, startOffsets};
}

// Device helpers similar to ones in main.cu
__device__ int d_deg(const int* __restrict__ flat, int loc) {
    int count = 0;
    while (true) {
        int v = flat[loc + count];
        if (v == INT_MIN || v == 0) break;
        ++count;
    }
    return count;
}

__device__ int d_con(const int* __restrict__ flat, int locA, int locB) {
    int i = locA;
    int j = locB;
    int count = 0;
    while (true) {
        int a = flat[i];
        int b = flat[j];
        if (a == INT_MIN || a == 0 || b == INT_MIN || b == 0) break;
        if (a == b) { ++count; ++i; ++j; }
        else if (a < b) { ++i; }
        else { ++j; }
    }
    return count;
}

__global__ void countType2PerHyperedgeKernel(const int* __restrict__ h2vStart,
                                             const int* __restrict__ h2vFlat,
                                             int h2vFixedSize,
                                             const int* __restrict__ h2hStart,
                                             const int* __restrict__ h2hFlat,
                                             int h2hFixedSize,
                                             long long* __restrict__ outCounts,
                                             int numHyperedges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numHyperedges) return;

    long long acc = 0;
    int startAdj = h2hStart[i];
    int pos = startAdj;
    // iterate neighbors of hyperedge i from adjacency payload
    while (pos < h2hFixedSize) {
        int neighborOneBased = h2hFlat[pos];
        if (neighborOneBased == INT_MIN || neighborOneBased == 0) break;
        int j = neighborOneBased - 1; // convert to 0-based index
        if (j > i) {
            // compute intersection size a and degrees
            int loc_i = h2vStart[i];
            int loc_j = h2vStart[j];
            int a = d_con(h2vFlat, loc_i, loc_j);
            if (a >= 2) {
                int deg_i = d_deg(h2vFlat, loc_i);
                int deg_j = d_deg(h2vFlat, loc_j);
                int b = deg_i - a;
                int c = deg_j - a;
                if (b > 0 && c > 0) {
                    long long aa = a;
                    acc += (aa * (aa - 1) / 2) * static_cast<long long>(b) * static_cast<long long>(c);
                }
            }
        }
        ++pos;
    }
    outCounts[i] = acc;
}

static inline void checkCudaLocal(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

int main(int argc, char* argv[]) {
    // Parameters: <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4]
    HypergraphParams params;
    if (!parseCommandLineArgs(argc, argv, params)) {
        return 1;
    }
    printHypergraphParams(params);

    // Generate host-side mappings
    auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);
    std::vector<std::vector<int>> hyperedge2hyperedge = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);

    // Flatten H2V and H2H locally
    auto [h2vFlatValues, h2vStartOffsets] = localFlatten2D(hyperedgeToVertex);
    auto [h2hFlatAdj,   h2hStartOffsets] = localFlatten2D(hyperedge2hyperedge);

    // Prepare keys [1..E]
    std::vector<int> keys(params.numHyperedges);
    for (int i = 0; i < params.numHyperedges; ++i) keys[i] = i + 1;

    // Build CBSTs for H2V and H2H
    CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
    h2vOps.construct(keys.data(), h2vStartOffsets.data(), params.numHyperedges,
                     h2vFlatValues.data(), static_cast<int>(h2vFlatValues.size()));

    CBSTOperations h2hOps("H2H", params.payloadCapacity, params.alignment);
    h2hOps.construct(keys.data(), h2hStartOffsets.data(), params.numHyperedges,
                     h2hFlatAdj.data(), static_cast<int>(h2hFlatAdj.size()));

    const CBSTContext& ctxH2V = h2vOps.context();
    const CBSTContext& ctxH2H = h2hOps.context();

    // Allocate output array on device
    long long* d_counts = nullptr;
    checkCudaLocal(cudaMalloc(&d_counts, params.numHyperedges * sizeof(long long)));
    checkCudaLocal(cudaMemset(d_counts, 0, params.numHyperedges * sizeof(long long)));

    // Launch kernel
    int block = 256;
    int grid = (params.numHyperedges + block - 1) / block;
    countType2PerHyperedgeKernel<<<grid, block>>>(
        ctxH2V.d_startOffsets,
        ctxH2V.d_flatPayload,
        ctxH2V.fixedSize,
        ctxH2H.d_startOffsets,
        ctxH2H.d_flatPayload,
        ctxH2H.fixedSize,
        d_counts,
        params.numHyperedges
    );
    checkCudaLocal(cudaDeviceSynchronize());

    // Copy back and sum on host
    std::vector<long long> counts(params.numHyperedges);
    checkCudaLocal(cudaMemcpy(counts.data(), d_counts, params.numHyperedges * sizeof(long long), cudaMemcpyDeviceToHost));
    long long total = 0;
    for (long long c : counts) total += c;
    std::cout << "[GPU] Type2 motif count: " << total << std::endl;

    checkCudaLocal(cudaFree(d_counts));
    return 0;
}



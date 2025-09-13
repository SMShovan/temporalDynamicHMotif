# Dynamic Hypergraph Analysis Project

A CUDA-based implementation for motif counting in dynamic hypergraphs using Complete Binary Search Trees and parallel processing.

## Project Overview

This project implements:
- **Hypergraph Data Structures**: Hyperedge-to-Vertex, Vertex-to-Hyperedge, and Hyperedge-to-Hyperedge mappings
- **Dynamic Data Management**: Complete Binary Search Trees on GPU for efficient insertion/deletion operations
- **Motif Counting**: Parallel counting of 30 different motif types in hypergraph triangles
- **GPU Acceleration**: CUDA kernels for high-performance computation

## Development Workflow

### Local Development + Cluster Execution

This project is designed for local development with cluster execution:

1. **Edit code locally** on your development machine
2. **Sync changes to cluster** using rsync
3. **Compile and run on cluster** with CUDA runtime

### Sync Commands

```bash
# Sync project to cluster
rsync -avz --progress /path/to/local/dynamicHyperGraph/ sskg8@mill.mst.edu:~/dynamicHyperGraph/

# Sync from cluster to local (if needed)
rsync -avz --progress sskg8@mill.mst.edu:~/dynamicHyperGraph/ /path/to/local/dynamicHyperGraph/
```

## Running on Cluster

### Load CUDA Module
```bash
module load cuda-toolkit/12.5
```

### Build and Run
```bash
nvcc main.cu -o main && ./main
```

## Project Structure

- `main.cu` - Main CUDA implementation with complete hypergraph processing pipeline
- `README.md` - This file

## Requirements

- CUDA Toolkit 12.5+
- NVIDIA GPU with CUDA support
- Thrust library (included with CUDA)

## Features

- **Dynamic Updates**: Real-time insertion and deletion of hyperedges
- **Memory Efficient**: Flattened array representations with GPU-optimized padding
- **Parallel Processing**: CUDA kernels for motif counting and tree operations
- **Scalable**: Designed for large hypergraph datasets

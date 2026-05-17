# ParallelComputingChallenge1# High-Performance CUDA Batched Matrix Multiplication

A highly optimized custom CUDA implementation for **Batched Arbitrary-Size Matrix Multiplication**. 

This project computes $P_i = M \times N_i$ for each $i \in \{0, ..., batch - 1\}$, where $M$ is a shared matrix across all multiplications, and $N_i$ are distinct matrices within a batch. 

## 🚀 Key Optimizations & Architecture
The custom kernel (`MatrixMulKernel2`) replaces the baseline standard matrix multiplication with advanced GPU memory optimization techniques to maximize bandwidth and compute throughput:

* **Shared Memory Tiling:** Data is loaded cooperatively by threads into block-level `__shared__` memory tiles to drastically reduce global memory accesses.
* **Register Tiling (Thread Coarsening):** Each thread calculates a $2 \times 2$ micro-tile within its private registers (`acc00` to `acc11`), increasing the arithmetic intensity and reducing memory latency.
* **Bank Conflict Avoidance:** Applied a $+1$ padding to the shared memory array dimensions (`[TILE_K + 1]`) to eliminate bank conflicts during memory fetches.
* **Loop Unrolling:** Used `#pragma unroll` to minimize loop overhead and improve instruction fetching at the assembly level.
* **Pointer Aliasing Restrictions:** Leveraged `__restrict__` pointers to inform the compiler that memory regions do not overlap, enabling further compiler-level optimizations.

## 🛠️ Build & Run Instructions

### Prerequisites
* NVIDIA GPU (Tested on architectures `sm_75` and newer)
* CUDA Toolkit

### Compilation
You can compile the project using the provided `Makefile`.
```bash
make all
```
### Run Instructions
Run following this example:
```bash
# Usage: ./bin/bmatmul <m> <k> <n> <batch> <seed>
./bin/bmatmul 1024 1024 1024 32 119
```
<m>: Rows of matrices $M$ and $P_i$<k>: Columns of $M$, Rows of $N_i$<n>: Columns of $N_i$ and $P_i$<batch>: Number of matrix pairs<seed>: Random initialization seed

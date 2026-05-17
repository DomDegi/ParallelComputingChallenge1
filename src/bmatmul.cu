// src/bmatmul.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Utility for wrapping CUDA API calls and logging errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// === Host-side validation ===
__host__
void batchedMatMulHost(float* M, float* N, float* P, int m, int k, int n, int batch) {
  for (int b = 0; b < batch; b++) {
    for (int row = 0; row < m; row++) {
      for (int col = 0; col < n; col++) {
        float value = 0.0f;
        for (int i = 0; i < k; i++) {
          float a = M[row*k + i];
          float c = N[b*(k*n) + i*n + col];
          value += a * c;
        }
        P[b*(m*n) + row*n + col] = value;
      }
    }
  }
}

void initWith(float number, float* arr, int size) {
  for (int i = 0; i < size; i++) arr[i] = number;
}

void initRandom(float* arr, int size, unsigned int seed, float minVal = 0.0f, float maxVal = 1.0f) {
  srand(seed);
  for (int i = 0; i < size; i++) {
    float r = (float)rand() / RAND_MAX;
    arr[i] = minVal + r * (maxVal - minVal);
  }
}

void checkResult(float* arr1, float* arr2, int size) {
  const float atol = 1e-4f; 
  const float rtol = 1e-4f; 
  for (int i = 0; i < size; i++) {
    float diff = fabs(arr1[i] - arr2[i]);
    float tol = atol + rtol*fabs(arr2[i]);
    if (diff > tol) {
      printf("Error at %d: %f != %f (diff=%e, tol=%e)\n", i, arr1[i], arr2[i], diff, tol);
      exit(1);
    }
  }
}

// === Device Kernels ===

// Tuning Parameters
#ifndef TILE_M
#define TILE_M 32   // Rows of M computed by a single block
#endif
#ifndef TILE_N
#define TILE_N 32   // Columns of the result computed by a single block
#endif
#ifndef TILE_K
#define TILE_K 16   // Elements processed at a time (inner dimension tile)
#endif
#ifndef RM
#define RM 2        // Rows per thread (register tile height)
#endif
#ifndef RN
#define RN 2        // Columns per thread (register tile width)
#endif

// Constraints: blockDim.y * RM == TILE_M and blockDim.x * RN == TILE_N
// With the values above: blockDim = (TILE_K, TILE_K) = (16,16).

/* * Optimized Batched Matrix Multiplication Kernel
 * * Explanation:
 * Each block computes a tile of size TILE_M x TILE_N.
 * Each thread computes a micro-tile of size RM x RN.
 * * We compute the starting row and column of the micro-tile using thread indices.
 * Data is fetched cooperatively into block shared memory (with padding to avoid bank conflicts).
 * Matrix multiplication is performed in registers using loop unrolling.
 * Finally, each thread writes its micro-tile back to the global memory result matrix Pb.
 */
__global__ void MatrixMulKernel2(const float* __restrict__ M,
                                 const float* __restrict__ N,
                                 float* __restrict__ P,
                                 int m, int k, int n, int batch)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int b  = blockIdx.z;
  
  // Boundary check for the batch dimension
  if (b >= batch) return;

  const int tx = threadIdx.x;   // in [0, TILE_K)
  const int ty = threadIdx.y;   // in [0, TILE_K)

  // Top-left corner of the thread's micro-tile (RM x RN)
  const int row0 = by * TILE_M + ty * RM;  
  const int col0 = bx * TILE_N + tx * RN;  

  // Batch Pointers: M is shared across all batches
  const float* __restrict__ Mb = M;                    
  const float* __restrict__ Nb = N + (size_t)b * k * n;
  float* __restrict__       Pb = P + (size_t)b * m * n;

  // Shared tiles (padding +1 to reduce bank conflicts)
  __shared__ float Ms[TILE_M][TILE_K + 1];
  __shared__ float Ns[TILE_K][TILE_N + 1];

  // Accumulators for the 2x2 micro-tile
  float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;

  const int numTilesK = (k + TILE_K - 1) / TILE_K;

  for (int tk = 0; tk < numTilesK; ++tk) {
    const int kBase = tk * TILE_K;

    // --- Cooperative fetch into shared memory: Ms ---
    #pragma unroll
    for (int r = 0; r < RM; ++r) {
      const int gr = row0 + r;
      const int gc = kBase + tx;
      Ms[ty * RM + r][tx] = (gr < m && gc < k) ? Mb[(size_t)gr * k + gc] : 0.f;
    }

    // --- Cooperative fetch into shared memory: Ns ---
    for (int kk = ty; kk < TILE_K; kk += blockDim.y) {
      int gr = kBase + kk;
      int gc = col0;

      float v0 = (gr < k && gc     < n) ? Nb[(size_t)gr * n + gc    ] : 0.f;
      float v1 = (gr < k && gc + 1 < n) ? Nb[(size_t)gr * n + gc + 1] : 0.f;

      Ns[kk][tx*RN + 0] = v0;
      Ns[kk][tx*RN + 1] = v1;
    }

    __syncthreads();

    // --- Matmul of the K panel ---
    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      // Fetch 2 rows from M
      const float a0 = Ms[ty * RM + 0][kk];
      const float a1 = Ms[ty * RM + 1][kk];

      // Fetch 2 columns from N
      const float b0 = Ns[kk][tx * RN + 0];
      const float b1 = Ns[kk][tx * RN + 1];

      // Compute partial products
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }

    __syncthreads();
  }

  // --- Write back the micro-tile to global memory ---
  if (row0 + 0 < m) {
    if (col0 + 0 < n) Pb[(size_t)(row0 + 0) * n + (col0 + 0)] = acc00;
    if (col0 + 1 < n) Pb[(size_t)(row0 + 0) * n + (col0 + 1)] = acc01;
  }
  if (row0 + 1 < m) {
    if (col0 + 0 < n) Pb[(size_t)(row0 + 1) * n + (col0 + 0)] = acc10;
    if (col0 + 1 < n) Pb[(size_t)(row0 + 1) * n + (col0 + 1)] = acc11;
  }
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf("Usage: %s <m> <k> <n> <batch> <seed>\n", argv[0]);
    exit(1);
  }

  int m = atoi(argv[1]); 
  int k = atoi(argv[2]); 
  int n = atoi(argv[3]); 
  int batch = atoi(argv[4]); 
  unsigned int seed = (unsigned int)atoi(argv[5]); 

  printf("Running batched matmul with m=%d, k=%d, n=%d, batch=%d, seed=%u\n", m, k, n, batch, seed);

  const int sizeM = m*k;
  const int sizeN = k*n*batch;
  const int sizeP = m*n*batch;

  float* M = (float*)malloc(sizeM * sizeof(float));
  float* N = (float*)malloc(sizeN * sizeof(float));
  float* P = (float*)malloc(sizeP * sizeof(float));

  initRandom(M, sizeM, seed);
  initRandom(N, sizeN, seed + 1);
  initWith(0.0f, P, sizeP);

  float *M_d, *N_d, *P_d;

  gpuErrchk(cudaMalloc((void**)&M_d, sizeM * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&N_d, sizeN * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&P_d, sizeP * sizeof(float)));

  gpuErrchk(cudaMemcpy(M_d, M, sizeM * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(N_d, N, sizeN * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(P_d, P, sizeP * sizeof(float), cudaMemcpyHostToDevice));

  // Grid and Block dimensions
  dim3 block(TILE_K, TILE_K); // (16,16)
  dim3 grid( (n + TILE_N - 1) / TILE_N,
             (m + TILE_M - 1) / TILE_M,
             batch );

  // Launch the optimized kernel
  MatrixMulKernel2<<<grid, block>>>(M_d, N_d, P_d, m, k, n, batch);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(P, P_d, sizeP * sizeof(float), cudaMemcpyDeviceToHost));

  printf("Checking results on CPU...\n");
  float* P_host = (float*)malloc(sizeP * sizeof(float));
  initWith(0.0f, P_host, sizeP);
  batchedMatMulHost(M, N, P_host, m, k, n, batch);
  checkResult(P, P_host, m*n*batch);
  printf("All results matched, success!\n");

  gpuErrchk(cudaFree(M_d));
  gpuErrchk(cudaFree(N_d));
  gpuErrchk(cudaFree(P_d));

  free(M); free(N); free(P); free(P_host);

  return 0;
}
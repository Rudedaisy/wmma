// -*- c++ -*-
// ^ The above line lets emacs visualize this as a c++ file

/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <curand.h>
#include <cuda_bf16.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// DCONV dimensions
const int N = 64;
const int C = 512;
const int C_offset = 32;//18;
//const int K = 512;
const int H = 64;
const int W = 64;
const int R = 3;
const int S = 3;
const int pad_h = 1;
const int pad_w = 1;
const int stride_h = 1;
const int stride_w = 1;
const int dilation_h = 1;
const int dilation_w = 1;

// Must be multiples of 16 for wmma code to work
const int MATRIX_M = N*H*W;
const int MATRIX_N = C_offset;
const int MATRIX_K = C*R*S;



// The only dimensions currently supported by WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__device__ void fused_conv2d_im2col_and_convert(__nv_bfloat16 *data_col,
						const float *data_im,
						const int n, const int warpM, const int warpN, const int tile_h, const int tile_w,
						const int N, const int C, const int H, const int W, const int R, const int S, const int P, const int Q,
						const int pad_h, const int pad_w,
						const int stride_h, const int stride_w,
						const int dilation_h, const int dilation_w
						) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % Q;
    const int h_col = (index / Q) % P;
    const int b_col = (index / Q / P) % N;
    const int c_im = (index / Q / P) / N;
    const int c_col = c_im * R * S;

    const int h_in = (warpM * tile_h) + (h_col * stride_h - pad_h);
    const int w_in = (warpN * tile_w) + (w_col * stride_w - pad_w);

    __nv_bfloat16 *data_col_ptr = data_col + ((c_col * N + b_col) * P + h_col) * Q + w_col;
    const float *data_im_ptr = data_im + (b_col * C + c_im) * H * W;

    for (int i = 0; i < R; ++i)
      for (int j = 0; j < S; ++j)
      {
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h;
        const float w_im = w_in + j * dilation_w;
	if (h_im > -1 && w_im > -1 && h_im < H && w_im < W){
          val = data_im_ptr[(int)h_im * (int)W + (int)w_im];
        } else {
          val = 0;
        }
        *data_col_ptr = __float2bfloat16(val);
        data_col_ptr += N * P * Q;
      }
  }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void stages1_2_gpu_kernel(float *offset,
				     float *in, __nv_bfloat16 *columns_in, __nv_bfloat16 *weight_offset, 
				     const int N, const int C, const int C_offset, const int H, const int W, const int R, const int S,
				     const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
				     const int MATRIX_M, const int MATRIX_N, const int MATRIX_K
				     ) {
   // Leading dimensions. Packed with no transpositions.
   int lda = MATRIX_M;
   int ldb = MATRIX_K;
   int ldc = MATRIX_M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Size of input matrix that needs to be
   int critical_unroll_n = warpSize * MATRIX_K;
   
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < MATRIX_K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N) {
	/*
	// im2col + convert input_fp32 to columns_input_bf16
	fused_conv2d_im2col_and_convert(columns_in,
					in,
					critical_unroll_n, warpM, warpN
					const int N, const int C, const int H, const int W, const int R, const int S, const int P, const int Q,
					const int pad_h, const int pad_w,
					const int stride_h, const int stride_w,
					const int dilation_h, const int dilation_w);
	*/
	// Load the inputs
	wmma::load_matrix_sync(a_frag, columns_in + aRow + aCol * lda, lda);
	wmma::load_matrix_sync(b_frag, weight_offset + bRow + bCol * ldb, ldb);
	
	// Perform the matrix multiplication
	wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	
      }
   }
   
   // Store result
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < MATRIX_M && cCol < MATRIX_N) {
     wmma::store_matrix_sync(offset + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
   }
}

__global__ void convertFp32ToFp16 (__nv_bfloat16 *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__global__ void convert_fp32_bf16 (__nv_bfloat16 *out, float *in, int n) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = __float2bfloat16(in[i]);
  }
}

int main(int argc, char* argv[]) {
   float *in_fp32;
   float *weight_offset_fp32;
   //   float *columns_in_fp32; ////////// REMOVE: USED TO POPULATE COLUMNS_IN WHILE CONVERT KERNEL UNDER CONSTRUCTION
   __nv_bfloat16 *columns_in_bf16;
   __nv_bfloat16 *weight_offset_bf16;
   float *offset_fp32;

   float *offset_fp32_host;
   
   curandGenerator_t gen;

   cudaEvent_t startConvertWeight;
   cudaEvent_t stopConvertWeight;
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;

   cudaErrCheck(cudaEventCreate(&startConvertWeight));
   cudaErrCheck(cudaEventCreate(&stopConvertWeight));
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   
   // Use tensor cores
   cudaErrCheck(cudaMalloc((void**)&in_fp32, N*H*W*C * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&weight_offset_fp32, C_offset*C*R*S * sizeof(float)));
   //   cudaErrCheck(cudaMalloc((void**)&columns_in_fp32, N*H*W*C*R*S * sizeof(float)));  /////////// REMOVE: USED TO POPULATE COLUMNS_IN WHILE CONVERT KERNEL UNDER CONSTRUCTION
   cudaErrCheck(cudaMalloc((void**)&columns_in_bf16, N*H*W*C*R*S * sizeof(__nv_bfloat16)));
   cudaErrCheck(cudaMalloc((void**)&weight_offset_bf16, C_offset*C*R*S * sizeof(__nv_bfloat16)));
   cudaErrCheck(cudaMalloc((void**)&offset_fp32, N*C_offset*H*W * sizeof(float)));
   
   
   offset_fp32_host = (float*)malloc(N*C_offset*H*W * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
   
   curandErrCheck(curandGenerateUniform(gen, in_fp32, N*C*H*W));
   curandErrCheck(curandGenerateUniform(gen, weight_offset_fp32, C_offset*C*R*S));
   //   curandErrCheck(curandGenerateUniform(gen, columns_in_fp32, N*H*W*C*R*S));  /////////// REMOVE: USED TO POPULATE COLUMNS_IN WHILE CONVERT KERNEL UNDER CONSTRUCTION

   //   convert_fp32_bf16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (columns_in_bf16, columns_in_fp32, N*H*W*C*R*S);  /////////// REMOVE: USED TO POPULATE COLUMNS_IN WHILE CONVERT KERNEL UNDER CONSTRUCTION
   
   curandErrCheck(curandDestroyGenerator(gen));

   dim3 gridDim;
   dim3 blockDim;
   
   // ***LAUNCH WEGIHT_OFFSET_FP32_BF16 KERNEL HERE*** //
   //blockDim.x = 256;
   //gridDim.x = (C_offset*C*R*S) / 256;
   cudaErrCheck(cudaEventRecord(startConvertWeight));
   convert_fp32_bf16 <<< (C_offset*C*R*S) / 256, 256>>> (weight_offset_bf16, weight_offset_fp32, C_offset*C*R*S);
   cudaErrCheck(cudaEventRecord(stopConvertWeight));
   
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   
   stages1_2_gpu_kernel <<< gridDim, blockDim >>> (offset_fp32,
						   in_fp32, columns_in_bf16, weight_offset_bf16,
						   N, C, C_offset, H, W, R, S,
						   pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
						   MATRIX_M, MATRIX_N, MATRIX_K); //////////////////////////////////////////
   cudaErrCheck(cudaEventRecord(stopWMMA));
   
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(offset_fp32_host, offset_fp32, N*C_offset*H*W * sizeof(float), cudaMemcpyDeviceToHost));
   printf("Offset result (top 5x5 section)");
   for (int i = 0; i < 5; i++) {
       for (int j = 0; j < 5; j++) {
       	   float v1 = offset_fp32_host[i*C_offset + j];
       	   printf("%f ", v1);
       }
       printf("\n");
   }

   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
   
   cudaErrCheck(cudaFree(in_fp32));
   cudaErrCheck(cudaFree(weight_offset_fp32));
   cudaErrCheck(cudaFree(columns_in_bf16));
   cudaErrCheck(cudaFree(weight_offset_bf16));
   cudaErrCheck(cudaFree(offset_fp32));
   
   free(offset_fp32_host);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}

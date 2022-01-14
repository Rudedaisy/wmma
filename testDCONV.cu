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
#include <assert.h>

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
const int P = (H + (2*pad_h) - 2*(int)(R/2)) / stride_h;
const int Q = (W + (2*pad_w) - 2*(int)(S/2)) / stride_w;

// Must be multiples of 16 for wmma code to work
const int MATRIX_M = N*H*W;
const int MATRIX_N = C_offset;
const int MATRIX_K = C*R*S;



// The only dimensions currently supported by WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CUDA_KERNEL_LOOP_X(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__device__ void fused_conv2d_im2col_and_convert(__nv_bfloat16 *data_col,
						const float *data_im,
						const int k, const int n, const int warpM, const int M_stride, const int warpM_index_stride,
						const int index, const int warpM_index, const int w_col, const int h_col, const int n_col, const int n_im, const int WMMA_SUB_TILE, __nv_bfloat16 **data_col_ptr, 
						const int N, const int C, const int H, const int W, const int R, const int S, const int P, const int Q,
						const int pad_h, const int pad_w,
						const int stride_h, const int stride_w,
						const int dilation_h, const int dilation_w
						) {
  // NOTE: current implementation has all warpN's repeating the same calculation if they are under the same warpM. May cause unwanted bandwidth pressure.
  int s_col;
  int r_col;
  int c_col;

  if (index < WMMA_SUB_TILE) {
    const int index_k = (index % WMMA_K) + k;
    s_col = index_k % S;
    r_col = (index_k / S) % R;
    c_col = (index_k / S / R) % C;

    const int s_displacement = (s_col - (int)(S / 2)) * dilation_w;
    const int r_displacement = (r_col - (int)(R / 2)) * dilation_h;

    const int c_im = c_col;
    const int w_im = (w_col * stride_w - pad_w) - s_displacement;
    const int h_im = (h_col * stride_h - pad_h) - r_displacement;

    //const float *data_im_ptr = data_im + (((n_im * C + c_im) * H + h_im) * W + w_im);
    const float *data_im_ptr = data_im + (((n_im * H + h_im) * W + w_im) * C + c_im);
    
    const int M_computed = (blockDim.x * blockDim.y) / WMMA_K;
    
#pragma unroll
    for(long long unsigned i_m = 0; i_m < (M_stride / M_computed); i_m++) { 
      // This if statement should not be divergent
      if (k == 0) {
	*data_col_ptr = data_col + (((((n_col * H + h_col) * W + w_col) * C + c_col) * R + r_col) * S + s_col);
      } else {
	*data_col_ptr += WMMA_K;
      }
      
      float val = static_cast<float>(0);
      if (h_im > -1 && w_im > -1 && h_im < H && w_im < W){
	val = *(data_im_ptr + i_m * M_computed * C);
      }
      assert((*data_col_ptr + i_m * warpM_index_stride) >= data_col);
      //assert((*data_col_ptr + i_m * warpM_index_stride) < (data_col + MATRIX_M*MATRIX_K));  ////////// ASSERT FAILS HERE, MUST CHECK
      *(*data_col_ptr + i_m * warpM_index_stride) = __float2bfloat16(val);
    }
  }
}

__device__ float conv2d_im2col_bilinear(const float *bottom_data, const int data_width,
					const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ void fused_conv2d_im2col_and_BLI(float *data_col,
					    const float *data_im, const float *data_offset,
					    const int k, const int n, const int warpM, const int M_stride, const int warpM_index_stride,
					    const int index, const int warpM_index, const int w_col, const int h_col, const int n_col, const int n_im, const int WMMA_SUB_TILE, float **data_col_ptr,
					    const int N, const int C, const int H, const int W, const int R, const int S, const int P, const int Q,
					    const int pad_h, const int pad_w,
					    const int stride_h, const int stride_w,
					    const int dilation_h, const int dilation_w
					    ) {
  // NOTE: current implementation has all warpN's repeating the same calculation if they are under the same warpM. May cause unwanted bandwidth pressure.
  int s_col;
  int r_col;
  int c_col;

  if (index < WMMA_SUB_TILE) {
    const int index_k = (index % WMMA_K) + k;
    s_col = index_k % S;
    r_col = (index_k / S) % R;
    c_col = (index_k / S / R) % C;

    const float *data_offset_ptr = data_offset + (((n_col * H + h_col) * W + w_col) * C_offset);
    const int offset_array_idx = 2 * (r_col * S + s_col);
    const float offset_h = data_offset_ptr[offset_array_idx];
    const float offset_w = data_offset_ptr[offset_array_idx + 1];
    
    const int s_displacement = (s_col - (int)(S / 2)) * dilation_w;
    const int r_displacement = (r_col - (int)(R / 2)) * dilation_h;

    const int c_im = c_col;
    const int w_im = (w_col * stride_w - pad_w) - s_displacement + offset_h; ////////////////////////////// CHECK ME: SHOULD USE OFFSET
    const int h_im = (h_col * stride_h - pad_h) - r_displacement + offset_w;

    const float *data_im_ptr = data_im + (((n_im * H + h_im) * W + w_im) * C + c_im);
    
    const int M_computed = (blockDim.x * blockDim.y) / WMMA_K;

#pragma unroll
    for(long long unsigned i_m = 0; i_m < (M_stride / M_computed); i_m++) {
      // This if statement should not be divergent
      if (k == 0) {
        *data_col_ptr = data_col + (((((n_col * H + h_col) * W + w_col) * C + c_col) * R + r_col) * S + s_col);
      } else {
        *data_col_ptr += WMMA_K;
      }

      float val = static_cast<float>(0);
      if (h_im > -1 && w_im > -1 && h_im < H && w_im < W){
        val = conv2d_im2col_bilinear((data_im_ptr + i_m * M_computed * C), W, H, W, h_im, w_im);
      }
      assert((*data_col_ptr + i_m * warpM_index_stride) >= data_col);
      //assert((*data_col_ptr + i_m * warpM_index_stride) < (data_col + MATRIX_M*MATRIX_K));
      *(*data_col_ptr + i_m * warpM_index_stride) = val;
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
				     const int N, const int C, const int C_offset, const int H, const int W, const int R, const int S, const int P, const int Q,
				     const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
				     const int MATRIX_M, const int MATRIX_N, const int MATRIX_K, const int M_stride, const int N_stride
				     ) {
   // Leading dimensions. Packed with no transpositions.
   int lda = MATRIX_M;
   int ldb = MATRIX_K;
   int ldc = MATRIX_M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fill_fragment(acc_frag, 0.0f);

   // Offset can now be statically allocated /// ED: In progress...
   //assert(M_stride==64 && N_stride==64);
   //float offsetsss[64*64];
   
   // ***Constant variables for im2col function***
   // In 2D block, the order of threads is [major=y, minor=x]
   const int index = (threadIdx.y * blockDim.x + threadIdx.x);
   //const int warpM_index = warpM * M_stride + (index / WMMA_K); // Incorrect, should multiply by WMMA_M
   const int warpM_index = warpM * WMMA_M + (index / WMMA_K); // IN PROGRESS
   assert((blockDim.x * blockDim.y) % WMMA_K == 0); // Need to be divisible for "warpM_index_stride" to be accurate
   const int warpM_index_stride = MATRIX_K * (int)((blockDim.x * blockDim.y) / WMMA_K);
   const int w_col = warpM_index % W;
   const int h_col = (warpM_index / W) % H;
   const int n_col = (warpM_index / W / H) % N;
   const int n_im = n_col;
   //const int WMMA_SUB_TILE = WMMA_M * WMMA_K;
   const int WMMA_SUB_TILE = M_stride * WMMA_K;
   __nv_bfloat16 *data_col_ptr;

   // Loop over k
#pragma	unroll
   for (int i = 0; i < MATRIX_K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N) {
	assert((aRow+WMMA_M) <= MATRIX_M && (aCol+WMMA_K) <= MATRIX_K && (bRow+WMMA_K) <= MATRIX_K && (bCol+WMMA_N) <= MATRIX_N); // Matrices need to be divisible by the WMMA dims
	
	// im2col + convert input_fp32 to columns_input_bf16
	fused_conv2d_im2col_and_convert(columns_in,
					in,
					i, MATRIX_K, warpM, M_stride, warpM_index_stride,
					index, warpM_index, w_col, h_col, n_col, n_im, WMMA_SUB_TILE, &data_col_ptr,
					N, C, H, W, R, S, P, Q,
					pad_h, pad_w,
					stride_h, stride_w,
					dilation_h, dilation_w);
	
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
     //wmma::store_matrix_sync(offsetsss, acc_frag, 64, wmma::mem_col_major);
   }

   // Perform BLI + unroll for main CONV
   
}

__global__ void convertFp32ToFp16 (__nv_bfloat16 *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__global__ void convert_fp32_bf16 (__nv_bfloat16 *out, float *in, int n) {
  CUDA_KERNEL_LOOP_X(i, n) {
    out[i] = __float2bfloat16(in[i]);
  }
}

int main(int argc, char* argv[]) {
   float *in_fp32;
   float *weight_offset_fp32;
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
   cudaErrCheck(cudaMalloc((void**)&columns_in_bf16, N*H*W*C*R*S * sizeof(__nv_bfloat16)));
   cudaErrCheck(cudaMalloc((void**)&weight_offset_bf16, C_offset*C*R*S * sizeof(__nv_bfloat16)));
   cudaErrCheck(cudaMalloc((void**)&offset_fp32, N*C_offset*H*W * sizeof(float)));
   
   
   offset_fp32_host = (float*)malloc(N*C_offset*H*W * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
   
   curandErrCheck(curandGenerateUniform(gen, in_fp32, N*C*H*W));
   curandErrCheck(curandGenerateUniform(gen, weight_offset_fp32, C_offset*C*R*S));
   
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
   /// ED: to transition to 4 warps, could try blockDim.x = 64, blockDim.y = 2
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   const int M_stride = (blockDim.x / 32) * WMMA_M;
   const int N_stride = (WMMA_N * blockDim.y);
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   
   stages1_2_gpu_kernel <<< gridDim, blockDim >>> (offset_fp32,
						   in_fp32, columns_in_bf16, weight_offset_bf16,
						   N, C, C_offset, H, W, R, S, P, Q,
						   pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
						   MATRIX_M, MATRIX_N, MATRIX_K, M_stride, N_stride); //////////////////////////////////////////
   cudaErrCheck(cudaEventRecord(stopWMMA));
   
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(offset_fp32_host, offset_fp32, N*C_offset*H*W * sizeof(float), cudaMemcpyDeviceToHost));
   printf("Offset result (top 5x5 section)\n");
   for (int i = 0; i < 20; i++) {
       for (int j = 0; j < MATRIX_N; j++) {
       	   float v1 = offset_fp32_host[i*MATRIX_M + j];
       	   printf("%.3f ", v1);
	   /*
	   if (v1 == 0) {
	     printf("Zero detected at idx=%d N/H/W = %d/%d/%d, K = %d\n", i, (i/W/H)%N, (i/W)%H, i%W, j);
	     //break;
	   }
	   //*/
       }
       printf("\n");
   }
   printf("Last: %f, +1: %f\n", offset_fp32_host[N*C_offset*H*W-1], offset_fp32_host[N*C_offset*H*W]);
   printf("MATRIX_M = %d\n", MATRIX_M);
   
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

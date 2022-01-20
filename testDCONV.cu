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
						const int k, const int blockM, const int blockM_index, const int M_stride, const int blockM_index_stride, const int tile_w,
						const int index, const int w_col, const int h_col, const int n_col, const int n_im, const int WMMA_SUB_TILE, __nv_bfloat16 **data_col_ptr,
						const int N, const int C, const int H, const int W, const int R, const int S, const int P, const int Q,
						const int pad_h, const int pad_w,
						const int stride_h, const int stride_w,
						const int dilation_h, const int dilation_w
						) {
  int s_col;
  int r_col;
  int c_col;

  if (index < WMMA_SUB_TILE) {
    const int index_k = (index % (WMMA_K*2)) + k;
    s_col = index_k % S;
    r_col = (index_k / S) % R;
    c_col = (index_k / S / R) % C;

    const int s_displacement = (s_col - (int)(S / 2)) * dilation_w;
    const int r_displacement = (r_col - (int)(R / 2)) * dilation_h;

    const int c_im = c_col;
    const int w_im = (w_col * stride_w - pad_w) - s_displacement;
    const int h_im = (h_col * stride_h - pad_h) - r_displacement;
    
    const float *data_im_ptr = data_im + (((n_im * H + h_im) * W + w_im) * C + c_im);
    
    const int M_computed = (blockDim.x * blockDim.y) / (WMMA_K*2);
    //assert(M_computed % tile_w == 0);
    //assert(M_computed >= tile_w);
    int H_tile_computed = M_computed / tile_w;
    const int tile_computed = H_tile_computed * W * C;

    // This if statement should not be divergent
    if (k == 0) {
      //*data_col_ptr = data_col + (((((n_col * H + h_col) * W + w_col) * C + c_col) * R + r_col) * S + s_col);
      //*data_col_ptr = data_col + (((((((n_col * nth + th) * ntw + tw) * tile_h + h_colx) * tile_w + w_colx) * C + c_col) * R + r_col) * S + s_col); // Actual shape: [N, H//tile_h, W//tile_w, tile_h, tile_w, C, R, S]
      *data_col_ptr = data_col + (((blockM_index * C + c_col) * R + r_col) * S + s_col); // Actual shape: [N, H//tile_h, W//tile_w, tile_h, tile_w, C, R, S]
    } else {
      *data_col_ptr += (WMMA_K*2);
    }
    
#pragma unroll
    for(long long unsigned i_m = 0; i_m < (M_stride / M_computed); i_m++) {
      float val = static_cast<float>(0);
      if (h_im > -1 && w_im > -1 && (h_im + i_m*H_tile_computed) < H && w_im < W){
	//assert((data_im_ptr + i_m * M_computed_stride * C) >= data_im);
	//assert((data_im_ptr + i_m * M_computed_stride * C) < data_im + MATRIX_M*C);
	//val = *(data_im_ptr + i_m * M_computed * C);
	val = *(data_im_ptr + i_m*tile_computed);
      }
      //assert((*data_col_ptr + i_m * blockM_index_stride) >= data_col);
      //assert((*data_col_ptr + i_m * blockM_index_stride) < (data_col + MATRIX_M*MATRIX_K));
      *(*data_col_ptr + i_m * blockM_index_stride) = __float2bfloat16(val);
    }
  }
}

__device__ float conv2d_im2col_bilinear(const float *bottom_data, const int data_width, const int minor_dim,
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
    v1 = bottom_data[(h_low * data_width + w_low) * minor_dim];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[(h_low * data_width + w_high) * minor_dim];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[(h_high * data_width + w_low) * minor_dim];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[(h_high * data_width + w_high) * minor_dim];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ void fused_conv2d_im2col_and_BLI(float *data_col,
					    const float *data_im, const float *data_offset_ptr,
					    const int k, const int blockM, const int M_stride, const int blockM_index_stride,
					    const int index, const int w_col, const int h_col, const int n_col, const int n_im, const int BLI_TILE, float **data_col_ptr,
					    const int N, const int C, const int H, const int W, const int R, const int S, const int P, const int Q,
					    const int pad_h, const int pad_w,
					    const int stride_h, const int stride_w,
					    const int dilation_h, const int dilation_w
					    ) {
  // NOTE: current implementation has all warpN's repeating the same calculation if they are under the same blockM. May cause unwanted bandwidth pressure.
  int s_col;
  int r_col;
  int c_col;

  if (index < BLI_TILE) {
    const int index_k = (index % WMMA_K) + k;
    s_col = index_k % S;
    r_col = (index_k / S) % R;
    c_col = (index_k / S / R) % C;

    const int offset_array_idx = 2 * (r_col * S + s_col);
    const float offset_h = data_offset_ptr[offset_array_idx];
    const float offset_w = data_offset_ptr[offset_array_idx + 1];
    
    const int s_displacement = (s_col - (int)(S / 2)) * dilation_w;
    const int r_displacement = (r_col - (int)(R / 2)) * dilation_h;

    const int c_im = c_col;
    const int w_im = (w_col * stride_w - pad_w) - s_displacement + offset_h;
    const int h_im = (h_col * stride_h - pad_h) - r_displacement + offset_w;

    const float *data_im_ptr = data_im + (((n_im * H + h_im) * W + w_im) * C + c_im);
    
    const int M_computed = (blockDim.x * blockDim.y) / WMMA_K;

    // This if statement should not be divergent
    if (k == 0) {
      *data_col_ptr = data_col + (((((n_col * H + h_col) * W + w_col) * C + c_col) * R + r_col) * S + s_col);
    } else {
      *data_col_ptr += WMMA_K;
    }

#pragma unroll
    for(long long unsigned i_m = 0; i_m < (M_stride / M_computed); i_m++) {
      float val = static_cast<float>(0);
      if (h_im > -1 && w_im > -1 && h_im < H && w_im < W){
        val = conv2d_im2col_bilinear((data_im_ptr + i_m * blockM_index_stride), W, C, H, W, h_im, w_im);
      }
      assert((*data_col_ptr + i_m * blockM_index_stride) >= data_col);
      //assert((*data_col_ptr + i_m * blockM_index_stride) < (data_col + MATRIX_M*MATRIX_K));
      *(*data_col_ptr + i_m * blockM_index_stride) = val;
    }
  }
}
      
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
__global__ void stages1_2_gpu_kernel(float *offset, float *deformed_columns_in,
				     float *in, __nv_bfloat16 *columns_in, __nv_bfloat16 *weight_offset, 
				     const int N, const int C, const int C_offset, const int H, const int W, const int R, const int S, const int P, const int Q, const int tile_h, const int tile_w,
				     const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
				     const int MATRIX_M, const int MATRIX_N, const int MATRIX_K, const int M_stride, const int N_stride
				     ) {
   // Leading dimensions. Packed with no transpositions.
   int lda = MATRIX_K;
   int ldb = MATRIX_K;
   int ldc = MATRIX_M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
   int blockM = blockIdx.x;
   
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fill_fragment(acc_frag, 0.0f);

   // Offset can now be statically allocated /// ED: In progress...
   //assert(M_stride==64 && N_stride==64);
   //__shared__ float offsetsss[128*32];
   
   // ***Constant variables for im2col function***
   // In 2D block, the order of threads is [major=y, minor=x]
   const int index = (threadIdx.y * blockDim.x + threadIdx.x);
   const int ntw = W / tile_w;
   const int nth = H / tile_h;
   const int blockM_index = blockM * M_stride + (index / (WMMA_K*2));
   //assert((blockDim.x * blockDim.y) % WMMA_K == 0); // Need to be divisible for "blockM_index_stride" to be accurate
   const int blockM_index_stride = MATRIX_K * (int)((blockDim.x * blockDim.y) / (WMMA_K*2));
   //const int w_col = blockM_index % W;
   //const int h_col = (blockM_index / W) % H;
   //const int n_col = (blockM_index / W / H) % N;
   const int w_colx = blockM_index % tile_w;
   const int h_colx = (blockM_index / tile_w) % tile_h;
   const int tw =     (blockM_index / tile_w / tile_h) % ntw;
   const int th =     (blockM_index / tile_w / tile_h / ntw) % nth;
   const int w_col = tw * tile_w + w_colx;
   const int h_col = th * tile_h + h_colx;
   const int n_col = (blockM_index / tile_w / tile_h / ntw / nth) % N;
   const int n_im = n_col;
   const int WMMA_SUB_TILE = M_stride * (WMMA_K*2);
   __nv_bfloat16 *data_col_ptr;

   //if (index == 0 && n_col==0) {
   //  printf("BlockM %d, batch/th/tw %d/%d/%d\n", blockM, n_col, th, tw);
   //}

   // Loop over k
#pragma	unroll
   for (int i = 0; i < MATRIX_K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N) {
	//assert((aRow+WMMA_M) <= MATRIX_M && (aCol+WMMA_K) <= MATRIX_K && (bRow+WMMA_K) <= MATRIX_K && (bCol+WMMA_N) <= MATRIX_N); // Matrices need to be divisible by the WMMA dims
	
	if (i % (WMMA_K*2) == 0) {
	// im2col + convert input_fp32 to columns_input_bf16
	  fused_conv2d_im2col_and_convert(columns_in,
					  in,
					  i, blockM, blockM_index, M_stride, blockM_index_stride, tile_w,
					  index, w_col, h_col, n_col, n_im, WMMA_SUB_TILE, &data_col_ptr,
					  N, C, H, W, R, S, P, Q,
					  pad_h, pad_w,
					  stride_h, stride_w,
					  dilation_h, dilation_w);
	  __syncthreads();
	}
	///*
	// Load the inputs
	wmma::load_matrix_sync(a_frag, columns_in + aRow + aCol * lda, lda);
	wmma::load_matrix_sync(b_frag, weight_offset + bRow + bCol * ldb, ldb);
	
	// Perform the matrix multiplication
	wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	//*/
      }
   }
   ///*
   // Store result
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;
   if (cRow < MATRIX_M && cCol < MATRIX_N) {
     wmma::store_matrix_sync(offset + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
     //wmma::store_matrix_sync(offsetsss, acc_frag, 128, wmma::mem_col_major);
   }
   
   // Perform BLI + unroll for main CONV
   const int BLI_TILE = M_stride * N_stride;
   //const float *data_offset_ptr = offset + (((n_col * H + h_col) * W + w_col) * C_offset); // C_offset = R*S*2
   const float *data_offset_ptr = offset + cRow + cCol * ldc;
   float *bli_data_col_ptr;
#pragma unroll
   for (int k = 0; k < MATRIX_K; k += WMMA_K) {
     fused_conv2d_im2col_and_BLI(deformed_columns_in,
				 in, data_offset_ptr,
				 k, blockM, M_stride, blockM_index_stride,
				 index, w_col, h_col, n_col, n_im, BLI_TILE, &bli_data_col_ptr,
				 N, C, H, W, R, S, P, Q,
				 pad_h, pad_w,
				 stride_h, stride_w,
				 dilation_h, dilation_w);
   }
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

float calculateSD(float data[], int len) {
  float sum = 0.0, mean, standardDeviation = 0.0;
  int i;

  for(i = 0; i < len; ++i) {
    sum += data[i];
  }

  mean = sum / len;

  for(i = 0; i < 10; ++i) {
    standardDeviation += pow(data[i] - mean, 2);
  }

  standardDeviation = sqrt(standardDeviation / len);

  printf("Mean %f, std %f\n", mean, standardDeviation);
  
  return standardDeviation;
}

int main(int argc, char* argv[]) {
   float *in_fp32;
   float *weight_offset_fp32;
   __nv_bfloat16 *columns_in_bf16;
   __nv_bfloat16 *weight_offset_bf16;
   float *offset_fp32;
   float *deformed_columns_in_fp32;

   __nv_bfloat16 *columns_in_bf16_host;
   float *offset_fp32_host;
   float *deformed_columns_in_fp32_host;
   
   curandGenerator_t gen;

   cudaEvent_t startConvertWeight;
   cudaEvent_t stopConvertWeight;
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;

   cudaErrCheck(cudaEventCreate(&startConvertWeight));
   cudaErrCheck(cudaEventCreate(&stopConvertWeight));
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));

   int tile_h = 16; // 16x8 is chosen to match the 128m warp
   int tile_w = 8;
   
   // Use tensor cores
   cudaErrCheck(cudaMalloc((void**)&in_fp32, N*H*W*C * sizeof(float)));                           
   cudaErrCheck(cudaMalloc((void**)&weight_offset_fp32, C_offset*C*R*S * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&columns_in_bf16, N*H*W*C*R*S * sizeof(__nv_bfloat16)));       // Actual shape: [N, H//tile_h, W//tile_w, tile_h, tile_w, C, R, S]
   cudaErrCheck(cudaMalloc((void**)&weight_offset_bf16, C_offset*C*R*S * sizeof(__nv_bfloat16)));
   cudaErrCheck(cudaMalloc((void**)&offset_fp32, N*C_offset*H*W * sizeof(float)));                // FUTURE: make temporary local array
   cudaErrCheck(cudaMalloc((void**)&deformed_columns_in_fp32, N*H*W*C*R*S * sizeof(float)));      // Actual shape: [N, H//tile_h, W//tile_w, tile_h, tile_w, C, R, S]      

   columns_in_bf16_host = (__nv_bfloat16*)malloc(N*H*W*C*R*S * sizeof(__nv_bfloat16));
   offset_fp32_host = (float*)malloc(N*C_offset*H*W * sizeof(float));
   deformed_columns_in_fp32_host = (float*)malloc(N*H*W*C*R*S * sizeof(float));
   
   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   //curandErrCheck(curandGenerateUniform(gen, in_fp32, N*C*H*W));                                                                                                                                                                                                          
   //curandErrCheck(curandGenerateUniform(gen, weight_offset_fp32, C_offset*C*R*S));
   curandErrCheck(curandGenerateNormal(gen, in_fp32, N*C*H*W, 0, 0.2));
   curandErrCheck(curandGenerateNormal(gen, weight_offset_fp32, C_offset*C*R*S, 0, 0.01));
   
   curandErrCheck(curandDestroyGenerator(gen));

   dim3 gridDim;
   dim3 blockDim;
   
   // ***LAUNCH WEGIHT_OFFSET_FP32_BF16 KERNEL HERE*** //
   //blockDim.x = 256;
   //gridDim.x = (C_offset*C*R*S) / 256;
   cudaErrCheck(cudaEventRecord(startConvertWeight));
   convert_fp32_bf16 <<< (C_offset*C*R*S) / 256, 256>>> (weight_offset_bf16, weight_offset_fp32, C_offset*C*R*S);
   cudaErrCheck(cudaEventRecord(stopConvertWeight));
   cudaErrCheck(cudaEventSynchronize(stopConvertWeight));

   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   /// ED: CANNOT do 64x64 tile since N<=32. Need to halve blockDim.y
   /// ED: to transition to 4 warps, could try blockDim.x = 64, blockDim.y = 2
   blockDim.x = 256; //128;
   blockDim.y = 2; // 4

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   const int M_stride = (blockDim.x / 32) * WMMA_M;
   const int N_stride = (WMMA_N * blockDim.y);
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   
   stages1_2_gpu_kernel <<< gridDim, blockDim >>> (offset_fp32, deformed_columns_in_fp32,
						   in_fp32, columns_in_bf16, weight_offset_bf16,
						   N, C, C_offset, H, W, R, S, P, Q, tile_h, tile_w,
						   pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
						   MATRIX_M, MATRIX_N, MATRIX_K, M_stride, N_stride);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   cudaErrCheck(cudaEventSynchronize(stopWMMA)); // Make sure event is done being recorded
   // check for error
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess) {
     // print the CUDA error message and exit
     printf("CUDA error: %s\n", cudaGetErrorString(error));
     exit(-1);
   }


   
   printf("\nPerformance timing:\n");
   float ms = 0;
   cudaErrCheck(cudaEventElapsedTime(&ms, startWMMA, stopWMMA));
   printf("\t%f ms\n", ms);

   
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(columns_in_bf16_host, columns_in_bf16, N*H*W*C*R*S * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(offset_fp32_host, offset_fp32, N*C_offset*H*W * sizeof(float), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(deformed_columns_in_fp32_host, deformed_columns_in_fp32, N*H*W*C*R*S * sizeof(float), cudaMemcpyDeviceToHost));

   /*
   printf("Unrolled input:\n");
   for (int i = 0; i < MATRIX_M; i++) {
     for (int j = 0; j < MATRIX_K; j++) {
       float val = __bfloat162float(columns_in_bf16_host[i*MATRIX_K + j]);
       //printf("%.3f ", val);
       if (val == 0) {
         printf("\nZero detected at idx=%d N/H/W = %d/%d/%d, C/R/S = %d/%d/%d\n", i, (i/W/H)%N, (i/W)%H, i%W, (j/R/S)%C, (j/S)%R, j%S);
	 // break;
       }
     }
     //printf("\n");
   }
   //*/
   
   /*
   unsigned reasonable = 0;
   unsigned unreasonable = 0;
   printf("Offset result\n");
   for (int i = 0; i < MATRIX_M; i++) {
       for (int j = 0; j < MATRIX_N; j++) {
       	   float v1 = offset_fp32_host[i*MATRIX_N + j];
       	   //printf("%.3f ", v1);
	   if (v1 == 0) {
	     printf("Zero detected at idx=%d N/H/W = %d/%d/%d, K = %d\n", i, (i/W/H)%N, (i/W)%H, i%W, j);
	     break;
	   }
	   if (v1 < 10) {
	     //printf("Reasonable offset detected at idx=%d N/H/W = %d/%d/%d, K = %d. VALUE = %f\n", i, (i/W/H)%N, (i/W)%H, i%W, j, v1);
	     reasonable++;
	   }
	   else {
	     //printf("*Unreasonable offset detected at idx=%d N/H/W = %d/%d/%d, K = %d. VALUE = %f\n", i, (i/W/H)%N, (i/W)%H, i%W, j, v1);
	     unreasonable++;
	   }
       }
       //printf("\n");
   }
   //printf("Last: %f, +1: %f\n", offset_fp32_host[N*C_offset*H*W-1], offset_fp32_host[N*C_offset*H*W]);
   //printf("MATRIX_M = %d\n", MATRIX_M);
   printf("Ratio of reasonable offsets: %f\n", (float)reasonable / (reasonable + unreasonable));
   calculateSD(offset_fp32_host, MATRIX_M*MATRIX_N);
   //*/

   /*
   printf("Deformed result:\n");
   for (int i = 0; i < MATRIX_M; i++) {
     for (int j = 0; j < MATRIX_K; j++) {
       //printf("%.3f ", deformed_columns_in_fp32_host[i*MATRIX_K + j]);
       if (deformed_columns_in_fp32_host[i*MATRIX_K + j] == 0) {
	 printf("\nZero detected at idx=%d N/H/W = %d/%d/%d, C/R/S = %d/%d/%d\n", i, (i/W/H)%N, (i/W)%H, i%W, (j/R/S)%C, (j/S)%R, j%S);
	 exit(0);
       }
     }
     //printf("\n");
   }
   //*/

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

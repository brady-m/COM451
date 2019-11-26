/**************************************************************************
*
*     set up GPU for processing
*
**************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include "gpu_main.h"

#include <cuda_runtime.h>

#define gScalar 0.2
texture<float, 2, cudaReadModeElementType> texBlue;

/******************************************************************************/
GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
  GPU_Palette X;

  X.gThreads.x = 32;  // 32 x 32 = 1024 threads per block
  X.gThreads.y = 32;
  X.gThreads.z = 1;
  X.gBlocks.x = ceil(imageWidth/32);  // however many blocks needed for image
  X.gBlocks.y = ceil(imageHeight/32);
  X.gBlocks.z = 1;

  X.palette_width = imageWidth;       // save this info
  X.palette_height = imageHeight;
  X.num_pixels = imageWidth * imageHeight;

  // allocate memory on GPU corresponding to pixel colors:
  cudaError_t err;
  err = cudaMalloc((void**) &X.red, X.num_pixels * sizeof(float));
  if(err != cudaSuccess){
    printf("cuda error allocating red = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  cudaMalloc((void**) &X.green, X.num_pixels * sizeof(float)); // g
  if(err != cudaSuccess){
    printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  cudaMalloc((void**) &X.blue, X.num_pixels * sizeof(float));  // b
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

	float *devPtr;
  size_t size= 64 * sizeof(float);

  cudaMalloc((void **) &devPtr, size);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  err = cudaBindTexture(NULL, &texBlue, devPtr, &channelDesc, size);

  if (err != cudaSuccess) {
      printf("cuda error bind texture = %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  return X;
}

/******************************************************************************/
void freeGPUPalette(GPU_Palette* P)
{
  cudaFree(P->red);
  cudaFree(P->green);
  cudaFree(P->blue);
}


/******************************************************************************/
int updatePalette(GPU_Palette* P, APoint (&points)[5])
// int updatePalette(GPU_Palette* P, int xIdx, int yIdx)
{
    for (int i = 0; i < 5; i++) {
        updateReds<<<P->gBlocks, P->gThreads>>>(P->red, points[i].xIdx, points[i].yIdx, points[i].z);
        updateGreens<<<P->gBlocks, P->gThreads>>>(P->green, points[i].xIdx, points[i].yIdx, points[i].z);
        updateBlues<<<P->gBlocks, P->gThreads>>>(P->blue, points[i].xIdx, points[i].yIdx, points[i].z);
    }
    return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, int xIdx, int yIdx, float z){
  int pS = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if ((xIdx < x + pS) && (xIdx > x - pS) && (yIdx < y + pS) && (yIdx > y - pS)) {
    red[vecIdx] = 1.0;
  } else {
    red[vecIdx] *= .98;
  }
}

/******************************************************************************/
__global__ void updateGreens(float* green, int xIdx, int yIdx, float z){
  int pS = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if ((xIdx < x + pS) && (xIdx > x - pS) && (yIdx < y + pS) && (yIdx > y - pS)) {
    green[vecIdx] = 1.0;
  } else {
    green[vecIdx] *= .90;
  }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, int xIdx, int yIdx, float z){
  int pS = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if ((xIdx < x + pS) && (xIdx > x - pS) && (yIdx < y + pS) && (yIdx > y - pS)) {
    blue[vecIdx] = 1.0;
  } else {
    float acc = 0.0;
    for (int i = -5;i <= 5;i++) {
      for (int j = -5;j <= 5;j++) {
        acc += tex2D(texBlue, x + i, y + j);
      }
    }
    acc /= 121.0;
    blue[vecIdx] = acc;
  }
}

/******************************************************************************/

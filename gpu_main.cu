/**************************************************************************
*
*     set up GPU for processing;
*
**************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include "gpu_main.h"

#include <cuda_runtime.h>

#define zInitialSize 3
#define zScale 1.1f
#define gScalar 0.2
texture<float, 2> texGreen;

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
  size_t size=64*sizeof(float);
  cudaMalloc((void **) &devPtr, size);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  err = cudaBindTexture2D(NULL, texGreen, X.green, channelDesc, imageWidth, imageHeight, sizeof(float) * imageWidth);
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
int updatePalette(GPU_Palette* P, int xIdx, int yIdx, float z)
{

  updateReds <<< P->gBlocks, P->gThreads >>> (P->red, xIdx, yIdx, z);
  updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, xIdx, yIdx, z);
	updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, xIdx, yIdx, z);

  return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, int xIdx, int yIdx, float z){

  int size = 5;
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {
    red[vecIdx] = 0.5;
  } else {
    red[vecIdx] *= .98;
  }
}

/******************************************************************************/
__global__ void updateGreens(float* green, int xIdx, int yIdx, float z){
  int size = 5;
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  float topPixel, leftPixel, centralPixel, rightPixel, bottomPixel;

  topPixel = tex2D(texGreen, x, y + 1);
  leftPixel = tex2D(texGreen, x - 1, y);
  centralPixel = tex2D(texGreen, x, y);
  rightPixel = tex2D(texGreen, x + 1, y);
  bottomPixel = tex2D(texGreen, x, y - 1);
  if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {
       green[vecIdx] = 0.8;
  } else {
      float meanHeat = (topPixel + bottomPixel + rightPixel + leftPixel + centralPixel) / (5 - 0.05);
      if (meanHeat >= 0.8) {
          green[vecIdx] = 0.8 / 4;
      } else {
          green[vecIdx] = meanHeat;
      }
      green[vecIdx] -= 0.01 * green[vecIdx];

      if (green[vecIdx] < 0.0)
          green[vecIdx] = 0.0;
      if (green[vecIdx] > 0.8)
          green[vecIdx] = 0.8;
  }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, int xIdx, int yIdx, float z){  
  int size = 5;
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {
    blue[vecIdx] = 1.0;
  } else {
    blue[vecIdx] *= .93;
  }
}

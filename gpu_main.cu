/**************************************************************************
*
*     set up GPU for processing
*
**************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include "gpu_main.h"
#include <cuda_runtime.h>
#include <math.h> 

#define gScalar 0.2

texture<float, 2> texRed;
texture<float, 2> texGreen;
texture<float, 2> texBlue;

/******************************************************************************/

GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
  GPU_Palette X;

  X.gThreads.x = 32;  // 32 x 32 = 1024 threads per block
  X.gThreads.y = 32;
  X.gThreads.z = 1;
  X.gBlocks.x = ceil(imageWidth / 32);  // however many blocks needed for image
  X.gBlocks.y = ceil(imageHeight / 32);
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


  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(NULL, texRed, X.red, desc, X.palette_width, X.palette_width, sizeof(float) * X.palette_width);
  cudaBindTexture2D(NULL, texGreen, X.green, desc, X.palette_width, X.palette_width, sizeof(float) * X.palette_width);
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, X.palette_width, X.palette_width, sizeof(float) * X.palette_width);


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

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if( (powf((x+5 - xIdx), 2) + powf((y+5 - yIdx), 2)) < powf(round(z*0.65), 2)) 
  {
    red[vecIdx] = 1;
  }
  else {
    red[vecIdx] *= 0.99;
  }
}

/******************************************************************************/

__global__ void updateGreens(float* green, int xIdx, int yIdx, float z){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if ((powf((x+5 - xIdx), 2) + powf((y+5 -yIdx), 2)) < powf(round(z*0.65), 2))
  {
    green[vecIdx] = 1.0;
  }
  else{
    green[vecIdx] *= .90;
  }
}

/******************************************************************************/

__global__ void updateBlues(float* blue, int xIdx, int yIdx, float z){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if((powf((x + 5 - xIdx), 2) + powf((y + 5 -yIdx), 2)) < powf(round(z * 0.65), 2))
  {
    blue[vecIdx] = 1.0;
  }
  else{
    blue[vecIdx] *= .90;
  }
}

/******************************************************************************/

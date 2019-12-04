/**************************************************************************
*
*     set up GPU for processing
*
**************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include "gpu_main.h"
#include <cuda_texture_types.h>
#include <cuda_runtime.h>
#include "PDP2_Anastasiya.h"

#define gScalar 0.4
texture<float, 2> texBlue;
texture<float, 2> texRed;
texture<float, 2> texGreen;
//define texBlue as texture memory here

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

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  err = cudaBindTexture2D(NULL, texBlue, X.blue, channelDesc, X.palette_width,
                          X.palette_width, sizeof(float) * X.palette_width);
  if (err != cudaSuccess) {
    printf("cuda error bind texture = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaBindTexture2D(NULL, texRed, X.red, channelDesc, X.palette_width,
                          X.palette_width, sizeof(float) * X.palette_width);
  if (err != cudaSuccess) {
    printf("cuda error bind texture = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaBindTexture2D(NULL, texGreen, X.green, channelDesc, X.palette_width,
                          X.palette_width, sizeof(float) * X.palette_width);
  if (err != cudaSuccess) {
    printf("cuda error bind texture = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // initialize texBlue here and bind to blue

  return X;
}

/******************************************************************************/
void freeGPUPalette(GPU_Palette* P)
{
  cudaUnbindTexture(texRed);
  cudaUnbindTexture(texGreen);
  cudaUnbindTexture(texBlue);
  cudaFree(P->red);
  cudaFree(P->green);
  cudaFree(P->blue);
}


/******************************************************************************/
int updatePalette(GPU_Palette* P, APoint (&points)[5])
{
  for (int i = 0;i < 5;i++) {
    updateReds <<< P->gBlocks, P->gThreads >>> (P->red, points[i].xIdx, points[i].yIdx, points[i].z, points[i].get_color());
    updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, points[i].xIdx, points[i].yIdx, points[i].z, points[i].get_color());
  	updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, points[i].xIdx, points[i].yIdx, points[i].z, points[i].get_color());
  }
  return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, int xIdx, int yIdx, float z, int c){

  int size = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  
  if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {
    red[vecIdx] = 1.0;
  } else if (c == Color::red){
        float top, left, center, right, bottom;
    float speed = 0.25;
    top = tex2D(texRed, x, y + 7);
    left = tex2D(texRed, x - 7, y);
    center = tex2D(texRed, x, y);
    right = tex2D(texRed, x + 7, y);
    bottom = tex2D(texRed, x, y - 7);
    red[vecIdx] = center + speed * (top + bottom + right + left - 4 * center);
  } else {
    red[vecIdx] *= 0.99;
  }
}

/******************************************************************************/
__global__ void updateGreens(float* green, int xIdx, int yIdx, float z, int c){
  int size = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {
    green[vecIdx] = 1.0;
  } else if (c == Color::green){
    float top, left, center, right, bottom;
    float speed = 0.25;
    top = tex2D(texRed, x, y + 7);
    left = tex2D(texRed, x - 7, y);
    center = tex2D(texRed, x, y);
    right = tex2D(texRed, x + 7, y);
    bottom = tex2D(texRed, x, y - 7);
    green[vecIdx] = center + speed * (top + bottom + right + left - 4 * center);
  } else {
    green[vecIdx] *= .99;
  }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, int xIdx, int yIdx, float z, int c){
  int size = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {
    blue[vecIdx] = 1.0;
  } else if (c == Color::blue){
    float top, left, center, right, bottom;
    float speed = 0.25;
    top = tex2D(texRed, x, y + 7);
    left = tex2D(texRed, x - 7, y);
    center = tex2D(texRed, x, y);
    right = tex2D(texRed, x + 7, y);
    bottom = tex2D(texRed, x, y - 7);
    blue[vecIdx] = center + speed * (top + bottom + right + left - 4 * center);
    // blue[vecIdx] = acc;
  } else {
    blue[vecIdx] *= .99;
  }
}
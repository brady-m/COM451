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
#include "drawing.h"

#define gScalar 0.2
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

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(NULL, texRed, X.red, desc, X.palette_width,
                    X.palette_width, sizeof(float) * X.palette_width);
  cudaBindTexture2D(NULL, texGreen, X.green, desc, X.palette_width,
                    X.palette_width, sizeof(float) * X.palette_width);
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, X.palette_width,
                    X.palette_width, sizeof(float) * X.palette_width);

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
    updateReds <<< P->gBlocks, P->gThreads >>> (P->red, points[i].xIdx, points[i].yIdx, points[i].z, points[i].color_heatTransfer);
    updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, points[i].xIdx, points[i].yIdx, points[i].z, points[i].color_heatTransfer);
  	updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, points[i].xIdx, points[i].yIdx, points[i].z, points[i].color_heatTransfer);
  }
  return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, int xIdx, int yIdx, float z, int colorTransfer){

  int pS = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(z*0.65);
  if ((powf((x - xIdx+5), 2) + powf((y - yIdx+5), 2)) < powf(pointSize, 2)) {
    red[vecIdx] = 0.6;
  } else if (colorTransfer == 0){
    float t, l, c, r, b;
    float speed = 0.25;
    t = tex2D(texRed, x, y - pointSize / 2);
    l = tex2D(texRed, x - pointSize / 2, y);
    c = tex2D(texRed, x, y) ;
    r = tex2D(texRed, x + pointSize / 2, y);
    b = tex2D(texRed, x, y + pointSize / 2);
    red[vecIdx] = c + speed * (t + b + r + l - 4 * c);
  } else {
    red[vecIdx] *= .99;
  }
}

/******************************************************************************/
__global__ void updateGreens(float* green, int xIdx, int yIdx, float z, int colorTransfer){
  int pS = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  int pointSize = round(z*0.65);
  if ((powf((x - xIdx+5), 2) + powf((y - yIdx+5), 2)) < powf(pointSize, 2)) {
    green[vecIdx] = 0.7;
  } else if (colorTransfer == 1){
    float t, l, c, r, b;
    float speed = 0.25;
    t = tex2D(texGreen,x,y-pointSize/2);
    l = tex2D(texGreen,x-pointSize/2,y);
    c = tex2D(texGreen,x,y);
    r = tex2D(texGreen,x+pointSize/2,y);
    b = tex2D(texGreen,x,y+pointSize/2);
    green[vecIdx] = c + speed * (t + b + r + l - 4 * c);
  } else {
    green[vecIdx] *= .99;
  }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, int xIdx, int yIdx, float z, int colorTransfer){
  int pS = (int)(5 + (z * gScalar));
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  int pointSize = round(z*0.65);
  if ((powf((x - xIdx+5), 2) + powf((y - yIdx+5), 2)) < powf(pointSize, 2)) {
    blue[vecIdx] = 0.9;
  } else if (colorTransfer == 2){
    float t, l, c, r, b;
    float speed = 0.25;
    t = tex2D(texBlue,x,y-pointSize/2);
    l = tex2D(texBlue,x-pointSize/2,y);
    c = tex2D(texBlue,x,y);
    r = tex2D(texBlue,x+pointSize/2,y);
    b = tex2D(texBlue,x,y+pointSize/2);
    blue[vecIdx] = c + speed * (t + b + r + l - 4 * c);
    // blue[vecIdx] = acc;
  } else {
    blue[vecIdx] *= .99;
  }
}

/******************************************************************************/

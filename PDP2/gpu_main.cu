/**************************************************************************
*
*     set up GPU for processing
*
**************************************************************************/

#include "gpu_main.h"
#include "interface.h"
#include <stdio.h>
#include <cuda_texture_types.h>

texture<float, 2> texBlue;

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
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, X.palette_width, 
                    X.palette_width, sizeof(float) * X.palette_width);

  return X;
}

/******************************************************************************/
void freeGPUPalette(GPU_Palette* P)
{
  cudaUnbindTexture(texBlue);

  cudaFree(P->red);
  cudaFree(P->green);
  cudaFree(P->blue);
}


/******************************************************************************/
int updatePalette(GPU_Palette* P, const Point& Point)
{
  // for (Point Point : Points.points) {
    updateReds   <<< P->gBlocks, P->gThreads >>> (P->red,   Point.xIdx, Point.yIdx, Point.z, Point.start_x);  // the randomness of color will be set 
    updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, Point.xIdx, Point.yIdx, Point.z, Point.start_y);  // based on starting coordinates of the Point
    updateBlues  <<< P->gBlocks, P->gThreads >>> (P->blue,  Point.xIdx, Point.yIdx, Point.z, Point.start_z);  // because they are initialized randomly
  // }
  return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, int xIdx, int yIdx, double z, double randColor){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(z*0.65);
          // x - xIdx+5 ???
  if( (powf((x+5 - xIdx), 2) + powf((y+5 - yIdx), 2)) < powf(pointSize, 2)) red[vecIdx] = randColor;
  else red[vecIdx] *= 0.9;
}

/******************************************************************************/
__global__ void updateGreens(float* green, int xIdx, int yIdx, double z, double randColor){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(z*0.65);

  if( (powf((x+5 - xIdx), 2) + powf((y+5 - yIdx), 2)) < powf(pointSize, 2)) green[vecIdx] = randColor;
  else green[vecIdx] *= 0.85;
}

/******************************************************************************/
__global__ void updateBlues(float* blue, int xIdx, int yIdx, double z, double randColor){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(z*0.65);

  if( (powf((x+5 - xIdx), 2) + powf((y+5 - yIdx), 2)) < powf(pointSize, 2)) blue[vecIdx] = randColor;
  else {
    float t, l, c, r, b;
    float speed = 0.25;
    t = tex2D(texBlue,x,y-pointSize/2);       
    l = tex2D(texBlue,x-pointSize/2,y);        
    c = tex2D(texBlue,x,y);        
    r = tex2D(texBlue,x+pointSize/2,y);        
    b = tex2D(texBlue,x,y+pointSize/2);      
    blue[vecIdx] = c + speed * (t + b + r + l - 4 * c);
    // blue[vecIdx] *= 0.99;
  }
}
/******************************************************************************/

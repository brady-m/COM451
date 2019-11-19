/**************************************************************************
*
*     set up GPU for processing
*
**************************************************************************/

#include "gpu_main.h"
#include <stdio.h>
#include <cuda_texture_types.h>

texture<float, 2> texRed;
texture<float, 2> texGreen;
texture<float, 2> texBlue;


GPU_Palette openPalette(int theWidth, int theHeight)
{
  unsigned long theSize = theWidth * theHeight;
  unsigned long memSize = theSize * sizeof(float);

  float* redmap = (float*) malloc(memSize);
  float* greenmap = (float*) malloc(memSize);
  float* bluemap = (float*) malloc(memSize);

  for(int i = 0; i < theSize; i++){
    bluemap[i] 	= .0;
    greenmap[i] = .0;
    redmap[i]   = .0;
  }

  GPU_Palette P1 = initGPUPalette(theWidth, theHeight);

  cudaMemcpy(P1.red, redmap, memSize, cH2D);
  cudaMemcpy(P1.green, greenmap, memSize, cH2D);
  cudaMemcpy(P1.blue, bluemap, memSize, cH2D);

  free(redmap);
  free(greenmap);
  free(bluemap);

  return P1;
}

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
int updatePalette(GPU_Palette* P, const Points& Points)
{
  for (Point Point : Points.points) {
    updateReds   <<< P->gBlocks, P->gThreads >>> (P->red,   Point);
    updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, Point);
    updateBlues  <<< P->gBlocks, P->gThreads >>> (P->blue,  Point);
  }
  return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, Point Point){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(Point.z*0.65);
          // x - xIdx+5 ???
  if( (powf((x+5 - Point.xIdx), 2) + powf((y+5 - Point.yIdx), 2)) < powf(pointSize, 2)) 
    red[vecIdx] = Point.red;
  else {
    if (Point.color_heatTransfer == 0) {
      float t, l, c, r, b;
      float speed = 0.25;
      t = tex2D(texRed,x,y-pointSize/2);       
      l = tex2D(texRed,x-pointSize/2,y);        
      c = tex2D(texRed,x,y);        
      r = tex2D(texRed,x+pointSize/2,y);        
      b = tex2D(texRed,x,y+pointSize/2);      
      red[vecIdx] = c + speed * (t + b + r + l - 4 * c);
    } 
    else {
      red[vecIdx] *= 0.99;
    }
  }
}

/******************************************************************************/
__global__ void updateGreens(float* green, Point Point){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(Point.z*0.65);

  if( (powf((x+5 - Point.xIdx), 2) + powf((y+5 - Point.yIdx), 2)) < powf(pointSize, 2)) 
    green[vecIdx] = Point.green;
  else {
    if (Point.color_heatTransfer == 1) {
      float t, l, c, r, b;
      float speed = 0.25;
      t = tex2D(texGreen,x,y-pointSize/2);       
      l = tex2D(texGreen,x-pointSize/2,y);        
      c = tex2D(texGreen,x,y);        
      r = tex2D(texGreen,x+pointSize/2,y);        
      b = tex2D(texGreen,x,y+pointSize/2);      
      green[vecIdx] = c + speed * (t + b + r + l - 4 * c);
    } 
    else {
      green[vecIdx] *= 0.99;
    }
  }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, Point Point){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  int pointSize = round(Point.z*0.65);

  if( (powf((x+5 - Point.xIdx), 2) + powf((y+5 - Point.yIdx), 2)) < powf(pointSize, 2)) 
    blue[vecIdx] = Point.blue;
  else {
    if (Point.color_heatTransfer == 2) {    
      float t, l, c, r, b;
      float speed = 0.25;
      t = tex2D(texBlue,x,y-pointSize/2);       
      l = tex2D(texBlue,x-pointSize/2,y);        
      c = tex2D(texBlue,x,y);        
      r = tex2D(texBlue,x+pointSize/2,y);        
      b = tex2D(texBlue,x,y+pointSize/2);      
      blue[vecIdx] = c + speed * (t + b + r + l - 4 * c);
    } 
    else {
      blue[vecIdx] *= 0.99;
    }
  }
}
/******************************************************************************/

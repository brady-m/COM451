#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand_kernel.h>          // has floor()
// #include "interface.h"

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct Point;

struct GPU_Palette{
  unsigned int palette_width;
  unsigned int palette_height;
  unsigned long num_pixels;

  dim3 gThreads;
  dim3 gBlocks;

  float* red;
  float* green;
  float* blue;
};

//GPU_Palette initGPUPalette(unsigned int, unsigned int);
GPU_Palette openPalette(int, int);
GPU_Palette initGPUPalette(unsigned int, unsigned int);
int updatePalette(GPU_Palette*, const Point&);
void freeGPUPalette(GPU_Palette*);

// kernel calls:
//__global__ void updateGrays(float* gray);
__global__ void updateReds(float* red, int, int, double, double);
__global__ void updateGreens(float* green, int, int, double, double);
__global__ void updateBlues(float* blue, int, int, double, double);
//__global__ void setup_rands(curandState* state, unsigned long seed, unsigned long);


#endif  // GPULib

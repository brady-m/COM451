#ifndef GPULib
#define GPULib
#include <cuda.h>
#include <curand_kernel.h>         
#include "point.h"

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

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


GPU_Palette openPalette(int, int);
GPU_Palette initGPUPalette(unsigned int, unsigned int);
int updatePalette(GPU_Palette*, APoint (&points)[5]);
void freeGPUPalette(GPU_Palette*);


__global__ void updateReds(float* red, int, int, float);
__global__ void updateGreens(float* green, int, int, float);
__global__ void updateBlues(float* blue, int, int, float);



#endif  

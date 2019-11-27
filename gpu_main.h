#ifndef GPULib
#define GPULib

#include <cuda.h>
//#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // has floor()
//#include <cuda_texture_types.h>
#include "point.h"
#define noa 10 //defines number of attractors, here separately because cant use from PDP2_Zh.h
#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost


struct APoint{
	float x;
	float y;
	float z;
	double xIdx, yIdx;
};

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
int updatePalette(GPU_Palette*, APoint (&points)[noa]);
//int updatePalette(GPU_Palette*, const APoint&);
void freeGPUPalette(GPU_Palette*);

// kernel calls:
//__global__ void updateGrays(float* gray);
__global__ void updateReds(float* red, int, int, float);
__global__ void updateGreens(float* green, int, int, float);
__global__ void updateBlues(float* blue, int, int, float);
//__global__ void setup_rands(curandState* state, unsigned long seed, unsigned long);


#endif  // GPULib

#ifndef GPULib
#define GPULib

#include "struct.h"

GPU_Palette openPalette(int, int);
GPU_Palette initGPUPalette(unsigned int, unsigned int);
int updatePalette(GPU_Palette*, const Points&);
void freeGPUPalette(GPU_Palette*);

// kernel calls:
//__global__ void updateGrays(float* gray);
__global__ void updateReds(float* red, const Point);
__global__ void updateGreens(float* green, const Point);
__global__ void updateBlues(float* blue, const Point);
//__global__ void setup_rands(curandState* state, unsigned long seed, unsigned long);


#endif  // GPULib

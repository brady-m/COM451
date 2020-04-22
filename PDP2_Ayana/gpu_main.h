#ifndef GPULib
#define GPULib

#include <cuda.h>
//#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // has floor()
//#include <cuda_texture_types.h>

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct GPU_Palette{

    unsigned int palette_width;
    unsigned int palette_height;
    unsigned long num_pixels;

    dim3 gThreads;
    dim3 gBlocks;

//    float* gray;
    float* red;
    float* green;
    float* blue;
//    float* dft;
//    curandState* rand;
};

//GPU_Palette initGPUPalette(unsigned int, unsigned int);
GPU_Palette openPalette(int, int);
GPU_Palette initGPUPalette(unsigned int, unsigned int);
int updatePalette(GPU_Palette*, int, int);
void freeGPUPalette(GPU_Palette*);

// kernel calls:
//__global__ void updateGrays(float* gray);
__global__ void updateReds(float* red, int, int);
__global__ void updateGreens(float* green, int, int);
__global__ void updateBlues(float* blue, int, int);
//__global__ void setup_rands(curandState* state, unsigned long seed, unsigned long);


#endif  // GPULib

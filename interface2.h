#ifndef hInterfaceLib
#define hInterfaceLib

//#include "params.h"
#include "gpu_main.h"
#include "animate.h"
//int runIt(GPU_Palette* P1, CPUAnimBitmap* A1);
//int runMode0(void);
//GPU_Palette openBMP(char* fileName);
//GPU_Palette openPalette(int, int);
//int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
//int usage();
int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
struct APoint{
	float x;
	float y;
	float z;
};
void printInformation();
void drawGraph();
// struct cudaDeviceProp {
//   char name[256];
//   size_t totalGlobalMem;
//   size_t sharedMemPerBlock;
//   int regsPerBlock;
//   int warpSize;
//   size_t memPitch;
//   int maxThreadsPerBlock;
//   int maxThreadsDim[3];
//   int maxGridSize[3];
//   size_t totalConstMem;
//   int major;
//   int minor;
//   int clockRate;
//   size_t textureAlignment;
//   int deviceOverlap;
//   int multiProcessorCount;
//   int kernelExecTimeoutEnabled;
//   int integrated;
//   int canMapHostMemory;
//   int computeMode;
//   int maxTexture1D;
//   int maxTexture2D[2];
//   int maxTexture3D[3];
//   int maxTexture2DArray[3];
//   int concurrentKernels;
// };
#endif

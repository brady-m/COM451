#ifndef STRUCTLib
#define STRUCTLib

#include <cuda.h>
#include <curand_kernel.h>          // has floor()

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

#define ITERATION_NUM     10000000              // iteration number
#define TEST_NUM          12                          // number for random starting points
#define NUMBER_OF_POINTS  5
#define gWIDTH            1920
#define gHEIGHT           1080

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

struct Point {
  double       x,       y,       z;
  double delta_x, delta_y, delta_z;
  double start_x, start_y, start_z;
  int xIdx, yIdx;
  double red,           blue,           green;
  int color_heatTransfer;
};

struct Points {
  Point points[NUMBER_OF_POINTS];
};

struct Parameters {
  bool  verbose;
  int   runMode;
  double a, b, c;
};

#endif

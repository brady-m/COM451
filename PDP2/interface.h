#ifndef hInterfaceLib
#define hInterfaceLib

#include "animate.h"
// #include "gpu_main.h"

#include <stdio.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <thread>
#include <iostream>

#define ITERATION_NUM     10000000              // iteration number
#define TEST_NUM          8                          // number for random starting points
#define NUMBER_OF_POINTS  1
#define gWIDTH            1920
#define gHEIGHT           1080

const double t = 0.005;

struct GPU_Palette;

struct Point {
  double       x,       y,       z;
  double delta_x, delta_y, delta_z;
  double start_x, start_y, start_z;
  int xIdx, yIdx;
};

struct Points {
  Point points[NUMBER_OF_POINTS];
};

struct Parameters {
  bool  verbose;
  int   runMode;
  double a, b, c;
};

double getRandNum();

/******************************RunMode 1*******************************************/
int getThreadNum(const std::string mode);
void calculateEquation(const std::string mode, Parameters& Parameters);
int runMode1(Parameters& Parameters);

/******************************RunMode 2*******************************************/
GPU_Palette openPalette(int theWidth, int theHeight);
int drawEquationuation(GPU_Palette* P1, CPUAnimBitmap* A1, const Parameters& Parameters, Point& Point);
int drawPal(GPU_Palette* P1, CPUAnimBitmap* A1);
int runMode2(const Parameters& Parameters, Points& Points);

#endif

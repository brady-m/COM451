
#include "gpu_main.h"
#include "animate.h"

int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);

struct AParams {
  bool  verbose;
  int   runMode;
  int   myParam1;
  float myParam2;
	double a, b, c;
};
void print_pc_information();
void drawGraph(const AParams& params);

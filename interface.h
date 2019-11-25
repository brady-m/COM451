#ifndef hInterfaceLib
#define hInterfaceLib

//#include "params.h"
#include "gpu_main.h"

//int runIt(GPU_Palette* P1, CPUAnimBitmap* A1);
//int runMode0(void);
//GPU_Palette openBMP(char* fileName);
//GPU_Palette openPalette(int, int);
//int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
//int usage();


// --- TODO: create interface.h library, move these there, and crack to here

struct AParams {
  bool  verbose;
  int   runMode;
  int   myParam1;
  float myParam2;
};

int usage();
int setDefaults(AParams *PARAMS);
int viewParams(const AParams *PARAMS);

void animateAttractor();

#endif

#ifndef hInterfaceLib
#define hInterfaceLib

#include <stdio.h>

#include "gpu_main.h"
#include "animate.h"

void drawAnimation();

// move these to interface.h library, and make class
int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
struct APoint{
	float x;
	float y;
	float z;
};


//int runMode0(void);
//GPU_Palette openBMP(char* fileName);
//GPU_Palette openPalette(int, int);
//int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
//int usage();

#endif

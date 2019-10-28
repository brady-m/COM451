#ifndef hInterfaceLib
	#define hInterfaceLib

	#include "gpu_main.h"
	#include "animate.h"

	int runIt(GPU_Palette* P1, CPUAnimBitmap* A1);
	struct APoint{
		float x;
		float y;
		float z;
	};
	void drawAttractor();
#endif

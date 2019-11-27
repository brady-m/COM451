/*******************************************************************************
*
*   Strange Attractor Base Code (assumes you have a gpu !)
*
*		crude way to get monitor size: xwininfo -root
*
*******************************************************************************/
#include <stdio.h>
#include <math.h>

#include "gpu_main.h"
#include "animate.h"
#include "interface.h"
#include "params.h"

// move these to interface.h library, and make class
int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
struct APoint{
	float x;
	float y;
	float z;
};

int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT


/******************************************************************************/
void drawAnimationPalette(){

	GPU_Palette P1;
	P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args

	CPUAnimBitmap animation(&P1);
	cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

	runIt(&P1, &animation);

	freeGPUPalette(&P1);

}

/******************************************************************************/
GPU_Palette openPalette(int theWidth, int theHeight)
{
	unsigned long theSize = theWidth * theHeight;

	unsigned long memSize = theSize * sizeof(float);
	float* redmap = (float*) malloc(memSize);
	float* greenmap = (float*) malloc(memSize);
	float* bluemap = (float*) malloc(memSize);

	for(int i = 0; i < theSize; i++){
		bluemap[i] 	= .7;
  		greenmap[i] = .2;
  		redmap[i]   = .2;
	}

	GPU_Palette P1 = initGPUPalette(theWidth, theHeight);

	cudaMemcpy(P1.red, redmap, memSize, cH2D);
	cudaMemcpy(P1.green, greenmap, memSize, cH2D);
	cudaMemcpy(P1.blue, bluemap, memSize, cH2D);

	free(redmap);
	free(greenmap);
	free(bluemap);

	return P1;
}

/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1){

	APoint thePoint, theChange; //theMins, theMaxs;
	float t = .005; // time step size


	thePoint.x = thePoint.y = thePoint.z = 0.5;

	int xIdx1, xIdx2, xIdx3, xIdx4, xIdx5;
	int yIdx1, yIdx2, yIdx3, yIdx4, yIdx5;

	for (long i = 1; i< 100000; i++)
	{

	static float a = 10.0;
	static float b = 28.0;
	static float c = 2.666;

	float dx, dy, dz;
	dx = t * a * cos(thePoint.y);
	dy = t * (b + thePoint.x - thePoint.z);
	dz = t * c * (thePoint.y - thePoint.z);

	thePoint.x += dx;
	thePoint.y += dy;
	thePoint.z += dz;

	static float minX = -20;
	static float maxX = 20;
	static float minY = -30;
	static float maxY = 30;

	xIdx1 = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
	yIdx1 = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)

	xIdx2 = floor((thePoint.x * 52) + 340);
	yIdx2 = floor((thePoint.y * 48) + 430);

	xIdx3 = floor((thePoint.x * 22) + 230);
	yIdx3 = floor((thePoint.y * 34) + 340);

	xIdx4 = floor((thePoint.x * 23) + 540);
	yIdx4 = floor((thePoint.y * 14) + 320);

	xIdx5 = floor((thePoint.x * 54) + 540);
	yIdx5 = floor((thePoint.y * 24) + 120);

	updatePalette(P1, xIdx1, yIdx1, thePoint.z);
	updatePalette(P1, xIdx2, yIdx2, thePoint.z);
	updatePalette(P1, xIdx3, yIdx3, thePoint.z);
	updatePalette(P1, xIdx4, yIdx4, thePoint.z);
	updatePalette(P1, xIdx5, yIdx5, thePoint.z);

  A1->drawPalette();
	}

return 0;
}

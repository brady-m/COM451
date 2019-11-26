/*******************************************************************************
*
*   Strange Attractor Base Code (assumes you have a gpu !)
*
*		crude way to get monitor size: xwininfo -root
*
*******************************************************************************/
#include <stdio.h>

#include "gpu_main.h"
#include "animate.h"

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
int drawAnimationPalette()
{

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

	float sigma = 10.0;
	float rho = 28.0;
	float beta = 2.666;

	int xIdx1, xIdx2, xIdx3, xIdx4, xIdx5;
	int yIdx1, yIdx2, yIdx3, yIdx4, yIdx5;

	for (long i = 1; i < 100000; i++)
	{
		theChange.x = t * sigma * cos(thePoint.y);
		theChange.y = t * (rho + thePoint.x - thePoint.z);
		theChange.z = t * beta * (thePoint.y - thePoint.z);

		thePoint.x += theChange.x;
		thePoint.y += theChange.y;
		thePoint.z += theChange.z;

		xIdx1 = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
		yIdx1 = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)

		xIdx2 = floor((thePoint.x * 32) + 360);
		yIdx2 = floor((thePoint.y * 18) + 140);

		xIdx3 = floor((thePoint.x * 32) + 540);
		yIdx3 = floor((thePoint.y * 18) + 230);

		xIdx4 = floor((thePoint.x * 32) + 630);
		yIdx4 = floor((thePoint.y * 18) + 410);

		xIdx5 = floor((thePoint.x * 32) + 320);
		yIdx5 = floor((thePoint.y * 18) + 760);

		updatePalette(P1, xIdx1, yIdx1);
		updatePalette(P1, xIdx2, yIdx2);
		updatePalette(P1, xIdx3, yIdx3);
		updatePalette(P1, xIdx4, yIdx4);
		updatePalette(P1, xIdx5, yIdx5);


    A1->drawPalette();

	}

return 0;
}

/******************************************************************************/

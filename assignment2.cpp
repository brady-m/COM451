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
void ass2(){

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
	float t = .05; // time step size
	thePoint.x = thePoint.y = thePoint.z = 0.5;

	float sigma = 1.0;
	float rho = 0.9;
	float beta = 0.4;

	int xIdx;
	int yIdx;

	for (long i = 1; i< 100000; i++)
	{

		//theChange.x = t * (sigma * (thePoint.y - thePoint.x));
		//theChange.y = t * ( (thePoint.x * (rho - thePoint.z)) - thePoint.y);
		//theChange.z = t * ( (thePoint.x * thePoint.y) - (beta * thePoint.z) );

		theChange.x = t * (sigma + (rho * ((thePoint.x*sin(thePoint.z))-(thePoint.y*cos(thePoint.z)))));
		theChange.y = t * (rho*((thePoint.x*cos(thePoint.z))+(thePoint.y*sin(thePoint.z))));
		theChange.z = t * (beta - (6.0/(1.0 + powf(thePoint.x, 2.0) + powf(thePoint.y, 2.0))));

		thePoint.x += theChange.x;
		thePoint.y += theChange.y;
		thePoint.z += theChange.z;

		xIdx = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
		yIdx = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)

		updatePalette(P1, xIdx, yIdx,thePoint.z);
    A1->drawPalette();

	}

return 0;
}

/******************************************************************************/

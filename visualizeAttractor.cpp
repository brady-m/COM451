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
#include "visualizeAttractor.h"
#include <cmath>
#include <vector>
// move these to interface.h library, and make class
int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
struct APoint{
	float x;
	float y;
	float z;
};

int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT
double startX= 0.5;
double startY= 0.5;
double startZ= 0.5;

/******************************************************************************/
int visualizeAttractor(std::vector<double> startingCoordinates){

	if(startingCoordinates.size() > 0){
	startX = startingCoordinates[0];
	startY = startingCoordinates[1];
	startZ = startingCoordinates[2];
	}

	GPU_Palette P1;
	P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args

	CPUAnimBitmap animation(&P1);
	cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
   	animation.initAnimation();
 	
	runIt(&P1, &animation);

	freeGPUPalette(&P1);
	return 0;
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
		bluemap[i] 	= .0;
  	greenmap[i] = .0;
  	redmap[i]   = .0;
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
	float t = 0.01; // time step size
	thePoint.x = startX;
	thePoint.y = startY; 
	thePoint.z = startZ;

	float sigma = 0.2;
	float rho = 0.2;
	float beta = 5.7;

	int xIdx;
	int yIdx;

	for (long i = 1; i< 100000; i++)
	{
		theChange.x = t * (-(thePoint.y + thePoint.z));
		theChange.y = t * (thePoint.x + sigma * thePoint.y);
		theChange.z = t * (rho + thePoint.x * thePoint.z - beta * thePoint.z);
	
		thePoint.x += theChange.x;
		thePoint.y += theChange.y;
		thePoint.z += theChange.z;

		xIdx = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
		yIdx = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)

		updatePalette(P1, xIdx, yIdx);
    	A1->drawPalette();

	}

	return 0;
}

/******************************************************************************/

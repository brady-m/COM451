#include "gpu_main.h"
#include "animate.h"
#include "visualizeAttractor.h"
#include <cmath>

#include <time.h>
#include <vector>
#include <thread>
#include <sched.h>
#include <math.h>

int drawAttractors(GPU_Palette* P1,  CPUAnimBitmap* A1);

int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT
int numberOfAttractors = 5;


int visualizeAttractor() {

	GPU_Palette P1;
	P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args

	CPUAnimBitmap animation(&P1);
	cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  	animation.initAnimation();
 	
	drawAttractors(&P1, &animation);

	freeGPUPalette(&P1);
	return 0;
}

GPU_Palette openPalette(int theWidth, int theHeight) {

	unsigned long theSize = theWidth * theHeight;
	unsigned long memSize = theSize * sizeof(float);

	float* redmap = (float*) malloc(memSize);
	float* greenmap = (float*) malloc(memSize);
	float* bluemap = (float*) malloc(memSize);

	for(int i = 0; i < theSize; i++) {
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

int drawAttractors(GPU_Palette* P1, CPUAnimBitmap* A1){

	APoint thePoint, theChange; //theMins, theMaxs;
	float t = 0.01; // time step size
	thePoint.x = thePoint.y = thePoint.z = 0.5;

	float a = 0.2;
	float b = 0.2;
	float c = 5.7;

	int xIdx[numberOfAttractors];
	int yIdx[numberOfAttractors];

	int rangeX[5];
	int rangeY[5];

 	srand (time(NULL));


	for (int i = 0; i < numberOfAttractors; i++) {
		rangeX[i] = rand() % (gWIDTH - 400);
		rangeY[i] = rand() % (gHEIGHT - 400); 
	}

	int numberOfFrames = 1000000;

	for (long i = 1; i < numberOfFrames; i++)	{

		theChange.x = t * (-(thePoint.y + thePoint.z));
		theChange.y = t * (thePoint.x + a * thePoint.y);
		theChange.z = t * (b + thePoint.x * thePoint.z - c * thePoint.z);


		thePoint.x += theChange.x;
		thePoint.y += theChange.y;
		thePoint.z += theChange.z;

		for (int j = 0; j < numberOfAttractors; j++) {
			xIdx[j] = floor((thePoint.x * 32) + rangeX[j]); // (X * scalar) + (gWidth/2)
			yIdx[j] = floor((thePoint.y * 16) + rangeY[j]); // (Y * scalar) + (gHeight/2)
		}
		
		for (int j = 0; j < numberOfAttractors; j++) {
			updatePalette(P1, xIdx[j], yIdx[j], thePoint.z, j);			
		}


		A1->drawPalette();
	}
return 0;
}

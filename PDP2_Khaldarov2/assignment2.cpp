/*******************************************************************************
*
*   Strange Attractor Base Code (assumes you have a gpu !)
*
*		crude way to get monitor size: xwininfo -root
*
*******************************************************************************/
#include <stdio.h>
#include <thread>
#include "gpu_main.h"
#include "animate.h"
#include "assignment2.h"

// move these to interface.h library, and make class
int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);
//struct APoint{
	//float x;
	//float y;
	//float z;
//};

int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT
#define ATTRACTORS 5

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
/*****************************************************************************/
double randomNum() {
    return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

/*****************************************************************************/
int firstInit(APoint& thePoint) {
    thePoint.x = randomNum();
    thePoint.y = randomNum();
    thePoint.z = randomNum();

    return 0;
}
/*****************************************************************************/
int equation(APoint& thePoint, APoint& theChange) {

		const double t = 0.05;
		float sigma = 1.0;
		float rho = 0.9;
		float beta = 0.4;

		//int xIdx[5];
		//int yIdx[5];
		//float zIdx;

		//for (long i = 1; i< 100000; i++)
		//{
			theChange.x = t * (sigma + (rho * ((thePoint.x*sin(thePoint.z))-(thePoint.y*cos(thePoint.z)))));
			theChange.y = t * (rho*((thePoint.x*cos(thePoint.z))+(thePoint.y*sin(thePoint.z))));
			theChange.z = t * (beta - (6.0/(1.0 + powf(thePoint.x, 2.0) + powf(thePoint.y, 2.0))));

			thePoint.x += theChange.x;
			thePoint.y += theChange.y;
			thePoint.z += theChange.z;


	  static float minX = -20.0;
	  static float maxX = 20.0;
	  static float minY = -30.0;
	  static float maxY = 30.0;

	  static float xRange = fabs(maxX - minX);
	  static float xScalar = 0.9 * (gWIDTH/xRange);

	  static float yRange = fabs(maxY - minY);
	  static float yScalar = 0.9 * (gHEIGHT/yRange);

	  thePoint.xIdx = round(xScalar * (thePoint.x - minX));
	  thePoint.yIdx = round(yScalar * (thePoint.y - minY));

    return 0;
	//}
}


/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1){

	APoint thePoint, theChange; //theMins, theMaxs;
	//float t = .05; // time step size
	APoint points[ATTRACTORS];
	APoint changes[ATTRACTORS];
	thePoint.x = thePoint.y = thePoint.z = 0.5;

		//xIdx = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
		//yIdx = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)

	//	updatePalette(P1, xIdx, yIdx,thePoint.z);
  //  A1->drawPalette();

	for(int i = 0; i < ATTRACTORS; i++){
			firstInit(points[i]);
	}

	//int xIdx[5];
	//int yIdx[5];
	//float zIdx;

	int rangeX[5];
	int rangeY[5];

	std::thread threads[ATTRACTORS];

	for(long i = 1; i< 100000; i++)
	{
		for (int i = 0; i < ATTRACTORS; i++)
		{
				threads[i] = std::thread(equation, std::ref(points[i]), std::ref(changes[i]));
		}

		for(int t_id = 0; t_id < ATTRACTORS; t_id++)
		{
				threads[t_id].join();
		}

		updatePalette(P1, points);
		A1->drawPalette();

	}
return 0;
}

/******************************************************************************/

/*******************************************************************************
*
*   Strange Attractor Base Code (assumes you have a gpu !)
*
*		crude way to get monitor size: xwininfo -root
*
*******************************************************************************/
#include <stdio.h>
#include <thread>
#include <time.h>
#include "gpu_main.h"
#include "animate.h"
#include "VisAssign2.h"

// move these to interface.h library, and make class;

int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT
#define ATTRACTORS 5
const float t = .005;



/******************************************************************************/
void visAssign() {

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

double randomNum() {
    return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

int firstInit(APoint& thePoint) {
    thePoint.x = randomNum();
    thePoint.y = randomNum();
    thePoint.z = randomNum();

    return 0;
}

int coor(APoint& thePoint, APoint& theChange) {

	const double t = 0.005;
		theChange.x = t * (theChange.x * (thePoint.y - thePoint.x));
		theChange.y = t * ((thePoint.x * (theChange.x - thePoint.z)) - thePoint.y);
	  theChange.z = t * ( (thePoint.x * thePoint.y) - (theChange.z * thePoint.z));

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
	// float sigma = 10.0;
	// float rho = 28.0;
	// float beta = 2.666;
	//
	// theChange.x = t * (sigma * (thePoint.y - thePoint.x));
	// theChange.y = t * ( (thePoint.x * (rho - thePoint.z)) - thePoint.y);
	// theChange.z = t * ( (thePoint.x * thePoint.y) - (beta * thePoint.z) );
	//
	//
  //   thePoint.x += theChange.x;
  //   thePoint.y += theChange.y;
  //   thePoint.z += theChange.z;
	//
  //   static float minX = -20;
  //   static float maxX = 20;
  //   static float minY = -30;
  //   static float maxY = 30;

    // static float xRange = fabs(maxX - minX);
    // static float xScalar = 0.9 * (gWIDTH/xRange);
		//
    // static float yRange = fabs(maxY - minY);
    // static float yScalar =  0.9 * (gHEIGHT/yRange);
		//

		//
    // thePoint.xIdx = floor((thePoint.y * 32)) * 5.0; // (X * scalar) + (gWidth/2)
    // thePoint.yIdx = floor((thePoint.x * 18)) * 5.0; // (Y * scalar) + (gHeight/2)

    return 0;
}


/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1){

	APoint thePoint, theChange; //theMins, theMaxs;
	APoint points[ATTRACTORS];
	APoint changes[ATTRACTORS];

	 // time step size
	thePoint.x = thePoint.y = thePoint.z = 0.5;


	for (int i = 0; i < ATTRACTORS; i++) {
			firstInit(points[i]);
	}

	int xIdx[5];
	int yIdx[5];
	float zIdx;

	int rangeX[5];
	int rangeY[5];

	std::thread threads[ATTRACTORS];

	for (long i = 1; i< 100000; i++)
	{

		// theChange.x = t * (sigma * (thePoint.y - thePoint.x));
		// theChange.y = t * ( (thePoint.x * (rho - thePoint.z)) - thePoint.y);
		// theChange.z = t * ( (thePoint.x * thePoint.y) - (beta * thePoint.z) );

		// thePoint.x += theChange.x;
		// thePoint.y += theChange.y;
		// thePoint.z += theChange.z;

		// xIdx = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
		// yIdx = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)
		//
		// xIdx1 = floor((thePoint.x * 30) + 90); // (X * scalar) + (gWidth/2)
		// yIdx1 = floor((thePoint.y * 16) + 54); // (Y * scalar) + (gHeight/2)
		//
		// xIdx2 = floor((thePoint.x * 28) + 820); // (X * scalar) + (gWidth/2)
		// yIdx2 = floor((thePoint.y * 14) + 320); // (Y * scalar) + (gHeight/2)
		//
		// xIdx3 = floor((thePoint.x * 26) + 210); // (X * scalar) + (gWidth/2)
		// yIdx3 = floor((thePoint.y * 12) + 450); // (Y * scalar) + (gHeight/2)
		//
		// xIdx4 = floor((thePoint.x * 24) + 330); // (X * scalar) + (gWidth/2)
		// yIdx4 = floor((thePoint.y * 10) + 540); // (Y * scalar) + (gHeight/2)
		//
		// updatePalette(P1, xIdx, yIdx, thePoint.z);
		// updatePalette(P1, xIdx1, yIdx1, thePoint.z);
		// updatePalette(P1, xIdx2, yIdx2, thePoint.z);
		// updatePalette(P1, xIdx3, yIdx3, thePoint.z);
		// updatePalette(P1, xIdx4, yIdx4, thePoint.z);
    // A1->drawPalette();

		for (int i = 0; i < ATTRACTORS; i++) {
				threads[i] = std::thread(coor, std::ref(points[i]), std::ref(changes[i]));
		}

		for (int t_id = 0; t_id < ATTRACTORS; t_id++) {
				threads[t_id].join();
		}

		updatePalette(P1, points);
		A1->drawPalette();

	}

return 0;
}

/******************************************************************************/

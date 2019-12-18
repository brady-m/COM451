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
#include "draw.h"
#include "interface.h"
#include <thread>
#include <math.h>

// move these to interface.h library, and make class
int runIt(GPU_Palette* P1,  CPUAnimBitmap* A1);

int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT

double randNum() {
	 return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
/******************************************************************************/
void Animdraw(){

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

void initPoints(APoint& point) {
		point.start_x = randNum();
		point.start_y = randNum();
		point.start_z = randNum();
		point.x = point.start_x * 20;
	 	point.y = point.start_y * 20;
		point.z = point.start_z * 20;

		point.red = point.start_x;
		point.green = point.start_y;
		point.blue = point.start_z;

		if((point.red >= point.green) && (point.red >= point.blue))
		 	point.color_heatTransfer = 0;
	 	else if (point.green >= point.blue)
		 	point.color_heatTransfer = 1;
	 	else
		 	point.color_heatTransfer = 2;
}


int attactor(APoint& thePoint, APoint& theChange) {

	const double t = 0.005;
	static float a = 10.0;
	static float b = 28.0;
	static float c = 2.666;

	theChange.x = t * (a * (thePoint.y - thePoint.x));
	theChange.y = t * ( (thePoint.x * (b - thePoint.z)) - thePoint.y);
	theChange.z = t * ( (thePoint.x * thePoint.y) - (c * thePoint.z) );

	thePoint.x += theChange.x;
	thePoint.y += theChange.y;
	thePoint.z += theChange.z;


	static float minX = -50;
	static float maxX = 50;
	static float minY = -150;
	static float maxY = 150;


	static float xRange = fabs(maxX - minX);
	static float xScalar = 0.7 * (gWIDTH/xRange);

	static float yRange = fabs(maxY - minY);
	static float yScalar = 0.7 * (gHEIGHT/yRange);

	thePoint.xIdx = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
	thePoint.yIdx = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)


	//thePoint.xIdx = round(xScalar * (thePoint.x - minX));
	//thePoint.yIdx = round(yScalar * (thePoint.y - minY));
	return 0;
}

/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1){

	APoint points[5];
	APoint changes[5];
	for (int i = 0;i < 5;i++) {
		initPoints(points[i]);
	}
	srand(time(NULL));
	std::thread threads[5];
	while(true)
	{
		for (int j = 0;j < 5;j++) {
			threads[j] = std::thread(attactor, std::ref(points[j]), std::ref(changes[j]));
		}

		for (int j = 0;j < 5;j++) {
			threads[j].join();
		}
		updatePalette(P1, points);
		
		A1->drawPalette();
	}
	

return 0;
}

/******************************************************************************/

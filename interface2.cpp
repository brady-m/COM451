/*******************************************************************************
*
*   Strange Attractor Base Code (assumes you have a gpu !)
*
*		crude way to get monitor size: xwininfo -root
*
*******************************************************************************/
#include <stdio.h>
#include <iostream>
#include "gpu_main.h"
#include "animate.h"
#include "interface2.h"
#include <thread>
#include "struct-point.h"

// move these to interface.h library, and make class


int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT


/******************************************************************************/
// int main(int argc, char *argv[]){
// 	return 0;
// }
double randNum() {
  return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

void printInformation() {
		cudaDeviceProp prop;
		int count;
		cudaGetDeviceCount(&count);
		for (int i = 0;i < count;i++) {
			cudaGetDeviceProperties(&prop, i);
			std::cout << "Name of GPU card: " << prop.name << std::endl;
			std::cout << "Total Global Memory of the GPU card: " << prop.totalGlobalMem << std::endl;
			std::cout << "Maximum amount of shared memory per block: " << prop.sharedMemPerBlock << std::endl;
			std::cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
			std::cout << "Maximum number of blocks first dimension of the grid: " << prop.maxThreadsDim[0] << std::endl;
			std::cout << "Maximum number of blocks second dimension of the grid: " << prop.maxThreadsDim[1] << std::endl;
			std::cout << "Maximum number of blocks third dimension of the grid: " << prop.maxThreadsDim[2] << std::endl;
			std::cout << "Amount of available constant memory: " << prop.totalConstMem << std::endl;
			std::cout << "Number of multiprocessors on the GPU card: " << prop.multiProcessorCount << std::endl;
		}
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


int calculatePoint(APoint& point, APoint& change) {
	const double t = 0.005;
	change.x = t * (change.x * (point.y - point.x));
	change.y = t * ((point.x * (change.x - point.z)) - point.y);
  change.z = t * ( (point.x * point.y) - (change.z * point.z));

  point.x += change.x;
  point.y += change.y;
  point.z += change.z;

  static float minX = -20.0;
  static float maxX = 20.0;
  static float minY = -30.0;
  static float maxY = 30.0;

  static float xRange = fabs(maxX - minX);
  static float xScalar = 0.9 * (gWIDTH/xRange);

  static float yRange = fabs(maxY - minY);
  static float yScalar = 0.9 * (gHEIGHT/yRange);

  point.xIdx = round(xScalar * (point.x - minX));
  point.yIdx = round(yScalar * (point.y - minY));
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
			threads[j] = std::thread(calculatePoint, std::ref(points[j]), std::ref(changes[j]));
		}

		for (int j = 0;j < 5;j++) {
			threads[j].join();
		}
		updatePalette(P1, points);
		A1->drawPalette();
	}

return 0;
}

void drawGraph(const AParams& paramters) {
	GPU_Palette P1;
	P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args
	CPUAnimBitmap animation(&P1);
	cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
	animation.initAnimation();
	runIt(&P1, &animation);
	freeGPUPalette(&P1);
}

/******************************************************************************/

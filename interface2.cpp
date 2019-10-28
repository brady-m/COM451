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
// move these to interface.h library, and make class


int gWIDTH = 1920;		// PALETTE WIDTH
int gHEIGHT = 1080;		// PALETTE HEIGHT


/******************************************************************************/
// int main(int argc, char *argv[]){
// 	return 0;
// }

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

void drawGraph() {
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


/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1){

	APoint thePoint, theChange; //theMins, theMaxs;
	float t = .005; // time step size
	thePoint.x = thePoint.y = thePoint.z = 0.5;

	float sigma = 10.0;
	float rho = 28.0;
	float beta = 2.666;

	int xIdx;
	int yIdx;

	for (long i = 1; i< 100000; i++)
	{

		theChange.x = t * (sigma * (thePoint.y - thePoint.x));
		theChange.y = t * ( (thePoint.x * (rho - thePoint.z)) - thePoint.y);
		theChange.z = t * ( (thePoint.x * thePoint.y) - (beta * thePoint.z) );

		thePoint.x += theChange.x;
		thePoint.y += theChange.y;
		thePoint.z += theChange.z;

		xIdx = floor((thePoint.x * 32) + 960); // (X * scalar) + (gWidth/2)
		yIdx = floor((thePoint.y * 18) + 540); // (Y * scalar) + (gHeight/2)

		updatePalette(P1, xIdx, yIdx, thePoint.z);
    A1->drawPalette();

	}

return 0;
}

/******************************************************************************/

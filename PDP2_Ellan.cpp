/*******************************************************************************
 *
 *   Strange Attractor Base Code (assumes you have a gpu !)
 *
 *		crude way to get monitor size: xwininfo -root
 *
 *******************************************************************************/
#include "PDP2_Ellan.h"
#include <stdio.h>
#include <iostream>
#include <thread>
#include "animate.h"
#include "gpu_main.h"
#include "APoint.h"

// move these to interface.h library, and make class

int gWIDTH = 1920;   // PALETTE WIDTH
int gHEIGHT = 1080;  // PALETTE HEIGHT

/******************************************************************************/
// int main(int argc, char *argv[]){
// 	return 0;
// }
double get_random_number() { return ((double)rand() / (RAND_MAX)); }

void print_pc_information() {
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Name of GPU card: " << prop.name << std::endl;
        std::cout << "Total Global Memory of the GPU card: "
                  << prop.totalGlobalMem << std::endl;
        std::cout << "Maximum amount of shared memory per block: "
                  << prop.sharedMemPerBlock << std::endl;
        std::cout << "Maximum number of threads per block: "
                  << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum number of blocks first dimension of the grid: "
                  << prop.maxThreadsDim[0] << std::endl;
        std::cout << "Maximum number of blocks second dimension of the grid: "
                  << prop.maxThreadsDim[1] << std::endl;
        std::cout << "Maximum number of blocks third dimension of the grid: "
                  << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Amount of available constant memory: "
                  << prop.totalConstMem << std::endl;
        std::cout << "Number of multiprocessors on the GPU card: "
                  << prop.multiProcessorCount << std::endl;
    }
}

/******************************************************************************/
GPU_Palette openPalette(int theWidth, int theHeight) {
    unsigned long theSize = theWidth * theHeight;

    unsigned long memSize = theSize * sizeof(float);
    float* redmap = (float*)malloc(memSize);
    float* greenmap = (float*)malloc(memSize);
    float* bluemap = (float*)malloc(memSize);

    for (int i = 0; i < theSize; i++) {
        bluemap[i] = .0;
        greenmap[i] = .0;
        redmap[i] = .0;
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

void init_points(APoint& point) {
	
    point.start_x = get_random_number();
    point.start_y = get_random_number();
    point.start_z = get_random_number();
	
    point.x = point.start_x * 20;
    point.y = point.start_y * 20;
    point.z = point.start_z * 20;

    point.red = point.start_x;
    point.green = point.start_y;
    point.blue = point.start_z;
}

int calculatePoint(APoint& point) {
    const double t = 0.005;
    point.changed_x = t * (point.changed_x * (point.y - point.x));
    point.changed_y = t * ((point.x * point.changed_y - point.y - point.x * point.z));
    point.changed_z = t * ((point.x * point.y) - (point.changed_z * point.z));

    point.updateParametrs();

    float minX = -20.0;
    float maxX = 20.0;
    float minY = -30.0;
    float maxY = 30.0;
    float xRange = fabs(maxX - minX);
    float xScalar = 0.6 * (gWIDTH / xRange);
    float yRange = fabs(maxY - minY);
    float yScalar = 0.6 * (gHEIGHT / yRange);

    point.xIdx = round(xScalar * (point.x - minX));
    point.yIdx = round(yScalar * (point.y - minY));
    return 0;
}

/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1) {
    APoint points[5];
    for (int i = 0; i < 5; i++) {
        init_points(points[i]);
    }
    srand(time(NULL));
    std::thread threads[5];
    while (true) {
        for (int i = 0; i < 5; i++) {
            threads[i] = std::thread(calculatePoint, std::ref(points[i]));
        }

        for (int j = 0; j < 5; j++) {
            threads[j].join();
        }
        updatePalette(P1, points);
        A1->drawPalette();
    }

    return 0;
}

void drawGraph(const AParams& paramters) {
    GPU_Palette P1;
    P1 = openPalette(gWIDTH, gHEIGHT);  // width, height of palette as args
    CPUAnimBitmap animation(&P1);
    cudaMalloc((void**)&animation.dev_bitmap, animation.image_size());
    animation.initAnimation();
    runIt(&P1, &animation);
    freeGPUPalette(&P1);
}

/******************************************************************************/

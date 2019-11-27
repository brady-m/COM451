/*******************************************************************************
*
*   Strange Attractor Base Code (assumes you have a gpu !)
*
*		crude way to get monitor size: xwininfo -root
*
*******************************************************************************/
#include <stdio.h>
// #include <vector>
#include <thread>
#include <time.h>

#include "animate.h"

#include "gpu_main.h"

#include "PDP2_Zhanboloti.h"




const float t = 0.02; // time step size


int gWIDTH = 1920; // PALETTE WIDTH
int gHEIGHT = 1080; // PALETTE HEIGHT

void drawAnimation()
{
    GPU_Palette P1;
    P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args

    CPUAnimBitmap animation(&P1);
    cudaMalloc((void**)&animation.dev_bitmap, animation.image_size());
    animation.initAnimation();

    runIt(&P1, &animation);

    freeGPUPalette(&P1);
}

/******************************************************************************/
GPU_Palette openPalette(int theWidth, int theHeight)
{
    unsigned long theSize = theWidth * theHeight;

    unsigned long memSize = theSize * sizeof(float);
    float* redmap = (float*)malloc(memSize);
    float* greenmap = (float*)malloc(memSize);
    float* bluemap = (float*)malloc(memSize);

    for (int i = 0; i < theSize; i++) {
        redmap[i] = 0;
        greenmap[i] = 0;
        bluemap[i] = 0;
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

double getRandNum() {
    return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

int attr(APoint& point, APoint& newP) {
 
    static float a = 0.95f;
    static float b = 0.7f;
    static float c = 0.6f;
    static float d = 3.5f;
    static float e = 0.25f;
    static float f = 0.1f;

    newP.x = t *((point.z-b)*point.x-d*point.y);
    newP.y = t *(d*point.x+(point.z-b)*point.y);
    newP.z = t *(c+a*point.z-(pow(point.z,3)/3)-(pow(point.x,2)+pow(point.y,2))*(1+e*point.z)+pow(f*point.z*point.x,3));
  //  printf("%f %f %f\n",point.x, point.y, point.z);
    point.x+=newP.x;
    point.y+=newP.y;
    point.z+=newP.z;

    static float minX= -10;
    static float maxX = 10;

    static float minY = -3.5f;
    static float maxY = 9.5f;

    static float xRange = fabs(maxX-minX);
    static float xScalar = 0.9*(gWIDTH/xRange);
    static float yRange = fabs(maxY-minY);
    static float yScalar = 0.9*(gHEIGHT/yRange);
    point.xIdx = round(xScalar*(point.x-minX));
    point.yIdx = round(yScalar*(point.y-minY));

    return 0;
}

int initRandStartPoints(APoint& point) {
    point.x = getRandNum();
    point.y = getRandNum();
    point.z = getRandNum();

    return 0;
}

/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1)
{
    srand(time(NULL));

    APoint point, newP; //theMins, theMaxs;
    APoint points[NUMBER_OF_ATTRACTORS];
    APoint changes[NUMBER_OF_ATTRACTORS];

    for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {
        initRandStartPoints(points[i]);
    }

    int xIdx[NUMBER_OF_ATTRACTORS];
    int yIdx[NUMBER_OF_ATTRACTORS];
    float zIdx;

    int rangeX[NUMBER_OF_ATTRACTORS];
    int rangeY[NUMBER_OF_ATTRACTORS];

    std::thread threads[NUMBER_OF_ATTRACTORS];


    for (long i = 1; i < 100000; i++){

        for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {
            threads[i] = std::thread(attr, std::ref(points[i]), std::ref(changes[i]));
        }

        for (int t_id = 0; t_id < NUMBER_OF_ATTRACTORS; t_id++) {
            threads[t_id].join();
        }
        
        updatePalette(P1, points);
        A1->drawPalette();
    }

    return 0;
}

/******************************************************************************/

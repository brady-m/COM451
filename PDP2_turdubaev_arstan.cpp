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

#include "PDP2_turdubaev_arstan.h"

// #define BackgroundRed 0.141
// #define BackgroundGreen 0.212
// #define BackgroundBlue 0.396

#define BackgroundRed 0.0f
#define BackgroundGreen 0.0f
#define BackgroundBlue 0.0f

#define AttractorRed 0.545
#define AttractorGreen 0.847
#define AttractorBlue 0.741

#define NUMBER_OF_ATTRACTORS 5

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
        redmap[i] = BackgroundRed;
        greenmap[i] = BackgroundGreen;
        bluemap[i] = BackgroundBlue;
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
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1)
{

    APoint thePoint, theChange; //theMins, theMaxs;
    float t = .5; // time step size
    thePoint.x = thePoint.y = thePoint.z = 0.5;

    // float sigma = 10.0;
    // float rho = 28.0;
    // float beta = 2.666;

    // int xIdx;
    // int yIdx;

    int xIdx[5];
    int yIdx[5];
    float zIdx;

    int rangeX[5];
    int rangeY[5];

    /* initialize random seed: */
    srand(time(NULL));

    std::thread threads[NUMBER_OF_ATTRACTORS];

    for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {
        rangeX[i] = (rand() % (gWIDTH - 400)) + 150;
        rangeY[i] = (rand() % (gHEIGHT - 300)) + 100;
    }

    for (long i = 1; i < 100000; i++) {

        // theChange.x = t * (sigma * (thePoint.y - thePoint.x));
        // theChange.y = t * ( (thePoint.x * (rho - thePoint.z)) - thePoint.y);
        // theChange.z = t * ( (thePoint.x * thePoint.y) - (beta * thePoint.z) );

        theChange.x = t * (sin(cos(thePoint.y)));
        theChange.y = t * (sin(sin(cos(0.96)) / thePoint.x));
        theChange.z = sin(thePoint.y - thePoint.z);

        thePoint.x += theChange.x;
        thePoint.y += theChange.y;
        thePoint.z += theChange.z;

        // printf("Z = %f\n");

        for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {

            xIdx[i] = floor((thePoint.y * 32) + rangeX[i]); // (X * scalar) + (gWidth/2)
            yIdx[i] = floor((thePoint.x * 18) + rangeY[i]); // (Y * scalar) + (gHeight/2)

            threads[i] = std::thread(updatePalette, P1, xIdx[i], yIdx[i], thePoint.z);
        }

        for (int t_id = 0; t_id < NUMBER_OF_ATTRACTORS; t_id++) {
            threads[t_id].join();
        }

        A1->drawPalette();
    }

    return 0;
}

/******************************************************************************/

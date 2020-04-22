#include <stdio.h>
#include <thread>
#include <math.h>
#include "gpu_main.h"
#include "animate.h"
#include "myVisAssign2.h"

int runIt(GPU_Palette* P1, CPUAnimBitmap* A1);

struct APoint{
    float x;
    float y;
    float z;
}

int gWIDTH = 1920;
int gHEIGHT = 1080;

/*****************************************************************************/

int myVisAssign2(){

        GPU_Palette P1;
        P1 = openPalette(gWIDTH, gHEIGHT);

        CPUAnimBitmap animation(&P1);
        cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
    	animation.initAnimation();

        runIt(&P1, &animation);

        freeGPUPalette(&P1);
}

/*****************************************************************************/

GPU_Palette openPalette(int theWidth, int theHeight){
        unsigned long theSize = theWidth * theHeight;

        unsigned long memSize = theSize * sizeof(float);
        float* redmap = (float*) malloc(memSize);
        float* greenmap = (float*) malloc(memSize);
        float* bluemap = (float*) malloc(memSize);

        for(int i = 0; i < theSize; i++){
                bluemap[i]      = .3;
                greenmap[i] = .0;
                redmap[i]   = .1;
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

int runIt(GPU_Palette* P1, CPUAnimBitmap* A1){

        APoint myPoint, myNewDis;
        
        float t = 0.01;
        
        myPoint.x = myPoint.y = myPoint.z = 0.5;

        int xIdx;
        int yIdx;

        for (long i = 1; i < 100000; i++){

                float a = 0.9;
                float b = 3.1;
                float c = 6.2;

                myNewDis.x = t * a * cos(myPoint.y);
                myNewDis.y = t * (b + myPoint.x - myPoint.z);
                myNewDis.z = t * (c * (myPoint.y - myPoint.z));

                myPoint.x += myNewDis.x;
                myPoint.y += myNewDis.y;
                myPoint.z += myNewDis.z;

                xIdx = floor((myPoint.x * 32) + 960);
                yIdx = floor((myPoint.y * 18) + 540);

                updatePalette(P1, xIdx, yIdx);
        A1->drawPalette();
        }
return 0;
}

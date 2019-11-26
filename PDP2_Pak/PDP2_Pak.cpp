#include <stdio.h>
#include <thread>
#include <time.h>

#include "animate.h"
#include "gpu_main.h"

#include "PDP2_Pak.h"


#define BackgroundRed 0.0f
#define BackgroundGreen 0.0f
#define BackgroundBlue 0.0f

#define AttractorRed 0.545
#define AttractorGreen 0.847
#define AttractorBlue 0.741

#define NUMBER_OF_ATTRACTORS 5
static float t = .009; 


int gWIDTH = 1920; 
int gHEIGHT = 1080; 

void animateAttractor()
{
    GPU_Palette P1;
    P1 = openPalette(gWIDTH, gHEIGHT); 

    CPUAnimBitmap animation(&P1);
    cudaMalloc((void**)&animation.dev_bitmap, animation.image_size());
    animation.initAnimation();

    runIt(&P1, &animation);

    freeGPUPalette(&P1);
}

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

double getRandNum() {
    return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

int blah(APoint& thePoint, APoint& theChange) {


	static float o = 0.2;
	static float b = 0.2;
	static float p = 10;

	theChange.x = -t * (thePoint.y + thePoint.z);
	theChange.y = t * (thePoint.x + o * thePoint.y);
	theChange.z = t * (b + thePoint.x * thePoint.z - p* thePoint.z);

    thePoint.x += theChange.x;
    thePoint.y += theChange.y;
    thePoint.z += theChange.z;

	static float minX = -20;
	static float maxX = 20;
	static float minY = -30;
	static float maxY = 30;

    static float xRange = fabs(maxX - minX);
    static float xScalar = 0.9 * (gWIDTH/xRange);

    static float yRange = fabs(maxY - minY);
    static float yScalar =  0.9 * (gHEIGHT/yRange);

    thePoint.xIdx = round(xScalar * (thePoint.x - minX));
    thePoint.yIdx = round(yScalar * (thePoint.y - minY));	


    return 0;
}

int initRandStartPoints(APoint& thePoint) {
    thePoint.x = getRandNum();
    thePoint.y = getRandNum();
    thePoint.z = getRandNum();

    return 0;
}


int runIt(GPU_Palette* P1, CPUAnimBitmap* A1)
{
    srand(time(NULL));

    APoint thePoint, theChange; 
    APoint points[NUMBER_OF_ATTRACTORS];
    APoint changes[NUMBER_OF_ATTRACTORS];

    for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {
        initRandStartPoints(points[i]);
    }

    int xIdx[5];
    int yIdx[5];
    float zIdx;

    int rangeX[5];
    int rangeY[5];

    

    std::thread threads[NUMBER_OF_ATTRACTORS];


    for (long i = 1; i < 100000; i++) {

        for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {
            threads[i] = std::thread(blah, std::ref(points[i]), std::ref(changes[i]));
        }

        for (int t_id = 0; t_id < NUMBER_OF_ATTRACTORS; t_id++) {
            threads[t_id].join();
        }
        
        updatePalette(P1, points);
        A1->drawPalette();
    }

    return 0;
}



#include <stdio.h>
#include <thread>
#include <time.h>

#include "animate.h"
#include "gpu_main.h"

#include "PDP2_nikolai.h"


#define RedColorBackground 0.1f
#define GreenColorBackground 0.1f
#define BlueColorBackground 0.1f

#define NumOfItter 100000
#define NUMBER_OF_ATTRACTORS 5
static float t = .005;


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

/******************************************************************************/
GPU_Palette openPalette(int theWidth, int theHeight)
{
    unsigned long theSize = theWidth * theHeight;

    unsigned long memSize = theSize * sizeof(float);
    float* redmap = (float*)malloc(memSize);
    float* greenmap = (float*)malloc(memSize);
    float* bluemap = (float*)malloc(memSize);

    for (int i = 0; i < theSize; i++) {
        redmap[i] = RedColorBackground;
        greenmap[i] = GreenColorBackground;
        bluemap[i] = BlueColorBackground;
    }

    GPU_Palette Pal = initGPUPalette(theWidth, theHeight);

    cudaMemcpy(Pal.red, redmap, memSize, cH2D);
    cudaMemcpy(Pal.green, greenmap, memSize, cH2D);
    cudaMemcpy(Pal.blue, bluemap, memSize, cH2D);

    free(redmap);
    free(greenmap);
    free(bluemap);

    return Pal;
}

double Random() {
    return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

int blah(APoint& thePoint, APoint& theChange) {



	float sigma = -100;
	float rho = 1.0;
	float beta = 0.3;

	theChange.x = t * sigma * cos(thePoint.y);
	theChange.y = t * (rho + thePoint.x- thePoint.z);
	theChange.z = t * beta * (thePoint.y - thePoint.z);

    thePoint.x += theChange.x;
    thePoint.y += theChange.y;
    thePoint.z += theChange.z;

	float xMinimum = -20;
	float xMaximum = 20;
	float yMinimum = -30;
	float yMaximum = 30;

    float xR = fabs(xMaximum - xMinimum);
    float xScl = 0.9 * (gWIDTH/xR);

    float yR = fabs(yMaximum - yMinimum);
    float yScl =  0.9 * (gHEIGHT/yR);

    thePoint.xIdx = round(xScl * (thePoint.x - xMinimum));
    thePoint.yIdx = round(yScl * (thePoint.y - yMinimum));


    return 0;
}

int RandomPoint(APoint& thePoint) {
    thePoint.x = Random();
    thePoint.y = Random();
    thePoint.z = Random();

    return 0;
}

/******************************************************************************/
int runIt(GPU_Palette* P1, CPUAnimBitmap* A1)
{
    srand(time(NULL));

    APoint thePoint, theChange;
    APoint points[NUMBER_OF_ATTRACTORS];
    APoint changes[NUMBER_OF_ATTRACTORS];

    for (int i = 0; i < NUMBER_OF_ATTRACTORS; i++) {
        RandomPoint(points[i]);
    }

    int xIdx[NUMBER_OF_ATTRACTORS];
    int yIdx[NUMBER_OF_ATTRACTORS];
    float zIdx;

    int rangeX[NUMBER_OF_ATTRACTORS];
    int rangeY[NUMBER_OF_ATTRACTORS];


    std::thread threads[NUMBER_OF_ATTRACTORS];


    for (long i = 1; i < NumOfItter; i++) {

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

/******************************************************************************/

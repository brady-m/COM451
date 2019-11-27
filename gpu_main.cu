/**************************************************************************
*
*     set up GPU for processing
*
**************************************************************************/

#include "gpu_main.h"
#include <cuda.h>
#include <stdio.h>

// #include <cuda_runtime.h>


#define aRed 0.99f
#define aGreen 0.99f
#define aBlue 0.99f

#define zInitialSize 2
#define zScale 1.0f
#define fade 0.01f
#define h 0.05f

texture<float, 2> texRed;
texture<float, 2> texGreen;
texture<float, 2> texBlue;

/******************************************************************************/
GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
    GPU_Palette X;

    X.gThreads.x = 32; // 32 x 32 = 1024 threads per block
    X.gThreads.y = 32;
    X.gThreads.z = 1;
    X.gBlocks.x = ceil(imageWidth / 32); // however many blocks ng++ -w -c interface.cpp $(F1) $(F2) $(F3) $(F4)eeded for image
    X.gBlocks.y = ceil(imageHeight / 32);
    X.gBlocks.z = 1;

    X.palette_width = imageWidth; // save this info
    X.palette_height = imageHeight;
    X.num_pixels = imageWidth * imageHeight;

    // allocate memory on GPU corresponding to pixel colors:
    cudaError_t err;
    err = cudaMalloc((void**)&X.red, X.num_pixels * sizeof(float));
    if (err != cudaSuccess) {
        printf("cuda error allocating red = %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**)&X.green, X.num_pixels * sizeof(float)); // g
    if (err != cudaSuccess) {
        printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**)&X.blue, X.num_pixels * sizeof(float)); // b
    if (err != cudaSuccess) {
        printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(NULL, texRed, X.red, desc, imageWidth, imageHeight, sizeof(float) * imageWidth);
    cudaBindTexture2D(NULL, texGreen, X.red, desc, imageWidth, imageHeight, sizeof(float) * imageWidth);
    cudaBindTexture2D(NULL, texBlue, X.red, desc, imageWidth, imageHeight, sizeof(float) * imageWidth);

    return X;
}

/******************************************************************************/
void freeGPUPalette(GPU_Palette* P)
{
    cudaUnbindTexture(texRed);
    cudaUnbindTexture(texGreen);
    cudaUnbindTexture(texBlue);

    cudaFree(P->red);
    cudaFree(P->green);
    cudaFree(P->blue);
}

/******************************************************************************/
int updatePalette(GPU_Palette* P, APoint (&points)[noa])
// int updatePalette(GPU_Palette* P, int xIdx, int yIdx)
{
    for (int i = 0; i < noa; i++) {
        updateReds<<<P->gBlocks, P->gThreads>>>(P->red, points[i].xIdx, points[i].yIdx, points[i].z);
        updateGreens<<<P->gBlocks, P->gThreads>>>(P->green, points[i].xIdx, points[i].yIdx, points[i].z);
        updateBlues<<<P->gBlocks, P->gThreads>>>(P->blue, points[i].xIdx, points[i].yIdx, points[i].z);
    }
    return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, int xIdx, int yIdx, float zIdx)
{

    // float size = 5 + (zIdx * 0.1);
    float size = zInitialSize + zIdx * zScale;
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int vecIdx = x + (y * blockDim.x * gridDim.x);

    float top, left, center, right, bot;

    top = tex2D(texRed, x, y + 1);
    left = tex2D(texRed, x - 1, y);
    center = tex2D(texRed, x, y);
    right = tex2D(texRed, x + 1, y);
    bot = tex2D(texRed, x, y - 1);



    if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {

        red[vecIdx] = aRed;

    } else {

        float heat_average = (top + bot + right + left + center) / 5 ;

        
        if (heat_average >= aRed) {
            red[vecIdx] = aRed / 2;
        } else {
            red[vecIdx] = heat_average;
        }

        

        red[vecIdx] -= fade * red[vecIdx];

        if (red[vecIdx] < 0)
            red[vecIdx] = 0;
        if (red[vecIdx] > aRed)
            red[vecIdx] = aRed;
    }
}

/******************************************************************************/
__global__ void updateGreens(float* green, int xIdx, int yIdx, float zIdx)
{

    float size = zInitialSize + zIdx * zScale;
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int vecIdx = x + (y * blockDim.x * gridDim.x);

    float top, left, center, right, bot;

    top = tex2D(texRed, x, y + 1);
    left = tex2D(texRed, x - 1, y);
    center = tex2D(texRed, x, y);
    right = tex2D(texRed, x + 1, y);
    bot = tex2D(texRed, x, y - 1);

    

    if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {

        green[vecIdx] = aGreen;

    } else {

        float heat_average = (top + bot + right + left + center) / 5;

        if (heat_average >= aGreen) {
            green[vecIdx] = aGreen / 2;
        } else {
            green[vecIdx] = heat_average;
        }


        green[vecIdx] -= fade * green[vecIdx];

        if (green[vecIdx] < 0)
            green[vecIdx] = 0;
        if (green[vecIdx] > aGreen)
            green[vecIdx] = aGreen;

    }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, int xIdx, int yIdx, float zIdx)
{

    float size = zInitialSize + zIdx * zScale;
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int vecIdx = x + (y * blockDim.x * gridDim.x);

    float top, left, center, right, bot;

    top = tex2D(texRed, x, y + 1);
    left = tex2D(texRed, x - 1, y);
    center = tex2D(texRed, x, y);
    right = tex2D(texRed, x + 1, y);
    bot = tex2D(texRed, x, y - 1);

    // blue[vecIdx] = center + fade * (top + bot + right + left - 4 * center);
    // blue[vecIdx] =(top + bot + right + left + center) / 5.0;

    if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {

        blue[vecIdx] = aBlue;

    } else {


         blue[vecIdx] -= fade * blue[vecIdx];

        if (blue[vecIdx] < 0)
            blue[vecIdx] = 0;
         if (blue[vecIdx] > aBlue)
             blue[vecIdx] = aBlue;
    }
}

/******************************************************************************/

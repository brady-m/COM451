#include "gpu_main.h"
#include <cuda.h>
#include <stdio.h>




#define BackgroundRed 0.0f
#define BackgroundGreen 0.0f
#define BackgroundBlue 0.0f



#define AttractorRed 0.9f
#define AttractorGreen 1.0f
#define AttractorBlue 0.05f

#define zInitialSize 7
#define zScale 0.01f
#define FadeSpeed 0.01f
#define HeatTransferSpeed 0.05f

texture<float, 2> texRed;
texture<float, 2> texGreen;
texture<float, 2> texBlue;


GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
    GPU_Palette X;

    X.gThreads.x = 32; 
    X.gThreads.y = 32;
    X.gThreads.z = 1;
    X.gBlocks.x = ceil(imageWidth / 32); 
    X.gBlocks.y = ceil(imageHeight / 32);
    X.gBlocks.z = 1;

    X.palette_width = imageWidth; 
    X.palette_height = imageHeight;
    X.num_pixels = imageWidth * imageHeight;

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


void freeGPUPalette(GPU_Palette* P)
{
    cudaUnbindTexture(texRed);
    cudaUnbindTexture(texGreen);
    cudaUnbindTexture(texBlue);

    cudaFree(P->red);
    cudaFree(P->green);
    cudaFree(P->blue);
}


int updatePalette(GPU_Palette* P, APoint (&points)[5])

{
    for (int i = 0; i < 5; i++) {
        updateReds<<<P->gBlocks, P->gThreads>>>(P->red, points[i].xIdx, points[i].yIdx, points[i].z);
        updateGreens<<<P->gBlocks, P->gThreads>>>(P->green, points[i].xIdx, points[i].yIdx, points[i].z);
        updateBlues<<<P->gBlocks, P->gThreads>>>(P->blue, points[i].xIdx, points[i].yIdx, points[i].z);
    }
    return 0;
}


__global__ void updateReds(float* red, int xIdx, int yIdx, float zIdx)
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

        red[vecIdx] = AttractorRed;

    } else {

        float heat_average = (top + bot + right + left + center) / (5 - HeatTransferSpeed);

       

        if (heat_average >= AttractorRed) {
            red[vecIdx] = AttractorRed / 2;
        } else {
            red[vecIdx] = heat_average;
        }

        red[vecIdx] -= FadeSpeed * red[vecIdx];

        if (red[vecIdx] < BackgroundRed)
            red[vecIdx] = BackgroundRed;
        if (red[vecIdx] > AttractorRed)
            red[vecIdx] = AttractorRed;
    }
}


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

        green[vecIdx] = AttractorGreen;

    } else {

        float heat_average = (top + bot + right + left + center) / (5 - HeatTransferSpeed);

        if (heat_average >= AttractorGreen) {
            green[vecIdx] = AttractorGreen / 2;
        } else {
            green[vecIdx] = heat_average;
        }


        green[vecIdx] -= FadeSpeed * green[vecIdx];

        if (green[vecIdx] < BackgroundGreen)
            green[vecIdx] = BackgroundGreen;
        if (green[vecIdx] > AttractorGreen)
            green[vecIdx] = AttractorGreen;

    }
}


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

   


    if (sqrtf(powf((x - xIdx), 2) + powf((y - yIdx), 2)) < size) {

        blue[vecIdx] = AttractorBlue;

    } else {

         blue[vecIdx] -= FadeSpeed * blue[vecIdx];

        if (blue[vecIdx] < BackgroundBlue)
            blue[vecIdx] = BackgroundBlue;
       
    }
}

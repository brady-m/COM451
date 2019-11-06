/*******************************************************************************
*
*   ANIMATION PROCESSING
*
*******************************************************************************/
#include "animate.h"
#include <stdio.h>

/******************************************************************************/
__global__ void drawColor(unsigned char* optr,
                          const float* red,
                          const float* green,
                          const float* blue) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float theRed = red[offset];
  if (theRed < 0) theRed = 0;
  if (theRed > 1) theRed = 1;

  float theGreen = green[offset];
  if (theGreen < 0) theGreen = 0;
  if (theGreen > 1) theGreen = 1;

  float theBlue = blue[offset];
  if (theBlue < 0) theBlue = 0;
  if (theBlue > 1) theBlue = 1;

  // convert RGB values from 0-1 to 0-255
  optr[offset * 4 + 0] = 255 * theRed;    // red
  optr[offset * 4 + 1] = 255 * theGreen;  // green
  optr[offset * 4 + 2] = 255 * theBlue;   // blue
  optr[offset * 4 + 3] = 255;             // alpha (opacity)
}

/******************************************************************************/
void CPUAnimBitmap::drawPalette(void) {

  dim3 threads(32, 32);
  dim3 blocks(ceil(width/32), ceil(height/32));

  drawColor <<< blocks, threads >>> (dev_bitmap,
                                     thePalette->red,
                                     thePalette->green,
                                     thePalette->blue);

  cudaMemcpy(get_ptr(), dev_bitmap, image_size(), cudaMemcpyDeviceToHost);
  glutMainLoopEvent();
  glutPostRedisplay();
}

/******************************************************************************/
CPUAnimBitmap::CPUAnimBitmap(GPU_Palette* P1) {
  width = P1->palette_width;
  height = P1->palette_height;
  pixels = new unsigned char[width * height * 4];

  thePalette = P1;
}

/******************************************************************************/
CPUAnimBitmap::~CPUAnimBitmap() {
  delete[] pixels;
}

/******************************************************************************/
CPUAnimBitmap** CPUAnimBitmap::get_bitmap_ptr(void) {
  static CPUAnimBitmap* gBitmap;
  return &gBitmap;
}

/******************************************************************************/
void CPUAnimBitmap::Draw(void) {
  CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE,
               bitmap->pixels);
  glutSwapBuffers();
}

/******************************************************************************/
void CPUAnimBitmap::initAnimation() {
  CPUAnimBitmap** bitmap = get_bitmap_ptr();
  *bitmap = this;
  int c = 1;
  char* dummy = "";
  glutInit(&c, &dummy);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(width, height);
  glutCreateWindow("MyWindow");
  glutDisplayFunc(Draw);
}

/******************************************************************************/

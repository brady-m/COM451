#include "interface.h"
#include "animate.h"
#include "gpu_main.h"

#include <stdio.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <thread>
#include <iostream>

double getRandNum() {
  return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

/******************************RunMode 1*******************************************/
int getThreadNum(const std::string mode)         // function to acquire number of cores on machine 
{                      
  return mode.compare("multi") == 0 ? get_nprocs() : 1;  // based on the chosen mode(multi or nonmulti)
}

void calculateEquation(const std::string mode, Parameters& Parameters) 
{
  Point Point;

  for (Parameters.a = 0.05; Parameters.a <= 1; Parameters.a+=0.05) {
    for (Parameters.b = 0.05; Parameters.b <= 1; Parameters.b+=0.05) {
      for (Parameters.c = 0.05; Parameters.c <= 1; Parameters.c+=0.05) {
                
        for (int i = 0; i < TEST_NUM/getThreadNum(mode); ++i) {
          Point.start_x = double(std::rand()) / (double(RAND_MAX) + 1.0);
          Point.start_y = double(std::rand()) / (double(RAND_MAX) + 1.0);
          Point.start_z = double(std::rand()) / (double(RAND_MAX) + 1.0);

          Point.x = Point.start_x;
          Point.y = Point.start_y;
          Point.z = Point.start_z;

          for (int j = 0; j < ITERATION_NUM; ++j) {
            Point.delta_x = t * (Parameters.a * (Point.y - Point.x));
            Point.delta_y = t * ( (Point.x * (Parameters.b - Point.z)) - Point.y);
            Point.delta_z = t * ( (Point.x * Point.y) - (Parameters.c * Point.z) );
            
            Point.x += Point.delta_x;
            Point.y += Point.delta_y;
            Point.z += Point.delta_z;
          }

          double totalChange = (fabs(Point.delta_x) + 
                                fabs(Point.delta_y) + 
                                fabs(Point.delta_z));
                    
          if (totalChange >= 0.01 &&
                Point.x <= 100 && Point.y <= 100 && Point.z <= 100) {
                    
            printf("\nPARAMETERS: sigma=%.2f, rho=%.2f, beta=%.2f\n", 
                    Parameters.a, Parameters.b, Parameters.c);
          }  else {
            printf("These parameters are invalid, searching for valid parameters\n");
          }
        }
      }
    }
  }
}

int runMode1(Parameters& Parameters)
{
  std::string mode;
  while (1) {

    printf("Choose mode (multi/nonmulti)\n");
    std::cin >> mode;

    if (mode.compare("exit") == 0) break;

    if (mode.compare("multi") == 0) {									// MULTI MODE

      printf("Number of cores: %d\n", getThreadNum(mode));
      std::thread threads[getThreadNum(mode)];

      for (int tId = 0; tId < getThreadNum(mode)-1; ++tId) {                    
        threads[tId] = std::thread(calculateEquation, mode, std::ref(Parameters));
      }

      calculateEquation(mode, Parameters);

      for (int tId = 0; tId < getThreadNum(mode)-1; ++tId) {
        threads[tId].join();
        // threads[tId].detach();
      }
    }
    else if (mode.compare("nonmulti") == 0) {                           // NONMULTI MODE
      calculateEquation(mode, Parameters);
    }
  }

  return 0;
}


/******************************RunMode 2*******************************************/

int initStartingPoints(Points& Points) {

  for (Point& Point : Points.points) {
    Point.start_x = getRandNum();
    Point.start_y = getRandNum();
    Point.start_z = getRandNum();

    Point.x = Point.start_x*10;
    Point.y = Point.start_y*10;
    Point.z = Point.start_z*10;

    Point.red = Point.start_x;
    Point.green = Point.start_y;
    Point.blue = Point.start_z;

    if((Point.red >= Point.green) && (Point.red >= Point.blue))
      Point.color_heatTransfer = 0;
    else if (Point.green >= Point.blue)
      Point.color_heatTransfer = 1;
    else
      Point.color_heatTransfer = 2;
  }

  return 0;
}

int drawEquation(const Parameters& Parameters, Point& Point) {

  Point.delta_x = t * (Parameters.a * (Point.y - Point.x));
  Point.delta_y = t * ( (Point.x * (Parameters.b - Point.z)) - Point.y);
  Point.delta_z = t * ( (Point.x * Point.y) - (Parameters.c * Point.z) );
  
  Point.x += Point.delta_x;
  Point.y += Point.delta_y;
  Point.z += Point.delta_z;

  static float minX = -20.0;
  static float maxX = 20.0;
  static float minY = -30.0;
  static float maxY = 30.0;

//   static float xRange = fabs(maxX - minX);
//   static float xScalar = 0.9 * (gWIDTH/xRange);

//   static float yRange = fabs(maxY - minY);
//   static float yScalar = 0.9 * (gHEIGHT/yRange);

//   Point.xIdx = round(xScalar * (Point.x - minX));
//   Point.yIdx = round(yScalar * (Point.y - minY));

  Point.xIdx = floor((Point.x * 32) + 960); // (X * scalar) + (gWidth/2)
  Point.yIdx = floor((Point.y * 18) + 540); // (Y * scalar) + (gHeight/2)
  
  return 0;
}

int runMode2(const Parameters& Parameters, Points& Points)  // ¯\_(ツ)_/¯
{
  GPU_Palette P1;
  P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args 

  CPUAnimBitmap animation(&P1);
  cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

  srand((unsigned)time(NULL));
  initStartingPoints(Points);

  std::thread threads[NUMBER_OF_POINTS];
  while (true) {

    for (int tId = 0; tId < NUMBER_OF_POINTS; ++tId) {
      threads[tId] = std::thread(drawEquation, std::ref(Parameters), std::ref(Points.points[tId]));
    }
    for (int tId = 0; tId < NUMBER_OF_POINTS; ++tId) {
      threads[tId].join();
    }
    updatePalette(&P1, Points);
    (&animation)->drawPalette();
  }

  freeGPUPalette(&P1);

  return 0;
}

#include <stdio.h>
#include <thread>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <unistd.h>
#include "PDP1_Li_Dima.h"

using namespace std;
int gNumThreads = 8;

struct Point {
  float x;
  float y;
  float z;
};

int start(Point p) {

	Point thePoint, thePoint2;
	float t = 0.005;

    thePoint.x = rand() % 1 + 0.5;
    thePoint.y = rand() % 1 + 0.5 + thePoint.x;
    thePoint.z = rand() % 1 + 0.1 + thePoint.y;

	float sigma = p.x;
	float rho = p.y;
	float beta = p.z;
	int xIdx;
	int yIdx;

	for (long i = 1; i< 100; i++) {
		thePoint2.x = t * sigma * cos(thePoint.y);
        thePoint2.y = t * (rho + thePoint.x - thePoint.z);
        thePoint2.z = t * beta * (thePoint.y - thePoint.z);

		thePoint.x += thePoint2.x;
		thePoint.y += thePoint2.y;
		thePoint.z += thePoint2.z;

		xIdx = floor((thePoint.x * 32) + 960);
		yIdx = floor((thePoint.y * 18) + 540);

		printf("x = %f, y = %f, z = %f\n", thePoint.x, thePoint.y, thePoint.z);

	}

return 0;
}

int run1() {
  int MULTITHREAD = 1;
  //int multithreadCheck;
  //cin >> multithreadCheck;

  time_t theStart, theEnd;
  time(&theStart);
  int procNum = sysconf(_SC_NPROCESSORS_ONLN);

  Point p;

  if(MULTITHREAD) {
      for (float a = 0.05; a <=1; a+=0.05){
          for (float b = 0.05; b <=1; b+=0.05){
              for (float c = 0.05; a <=1; a+=0.05) {

                p.x = a;
                p.y = b;
                p.z = c;

                thread zThreads[gNumThreads];

                for(int tid=0; tid < gNumThreads-1; tid++){
                  zThreads[tid] = thread(start, p);
                }

                for(int tid=0; tid<gNumThreads-1; tid++) {
                  zThreads[tid].join();
                }

              }
          }
      }
  } else {
	  for (int tid = 0; tid<8; tid++)
	  {
		  start(p);
	  }
  }
time(&theEnd);
if(MULTITHREAD)
  printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
else
  printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);
  printf(" %d Processes quantity\n", procNum);
return 0;

}

#include <stdio.h>
#include <thread>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <unistd.h>
#include <bits/stdc++.h>
#include "PDP2_Khaldarov.h"

using namespace std;

int gNumThreads = 8;
int itterations = 100;
//float timeS=0.001

struct Point
{
  float x,y,z,sigma,rho,beta;
};

int runIt(Point p)
{

	Point point1, point2;
	float timeS = 0.05;

    point1.x = rand() % 1 + 0.5;
    point1.y = rand() % 1 + 0.5 + point1.x;
    point1.z = rand() % 1 + 0.1 + point1.y;

	float sigma = p.x;
	float rho = p.y;
	float beta = p.z;

	for (int i = 0; i < itterations; i++)
	{
		    //point2.x = timeS * sigma * cos(point1.y);
        //point2.y = timeS * (rho + point1.x - point1.z);
        //point2.z = timeS * beta * (point1.y - point1.z);

        point2.x = timeS * (sigma + (rho * ((point1.x*sin(point1.z))-(point1.y*cos(point1.z)))));
        point2.y = timeS * (rho*((point1.x*cos(point1.z))+(point1.y*sin(point1.z))));
        point2.z = timeS * (beta - (6.0/(1.0 + powf(point1.x, 2.0) + powf(point1.y, 2.0))));

		point1.x += point2.x;
		point1.y += point2.y;
		point1.z += point2.z;

		printf("x = %f, y = %f, z = %f\n", point1.x, point1.y, point1.z);

	}
	return 0;
}

int assignment1()
{
  int MULTITHREAD = 1;
  time_t theStart, theEnd;
  time(&theStart);
  long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);

  Point p;

  if(MULTITHREAD)
  {
      for (float a = 0.05; a <=1; a+=0.05)
	  {
          for (float b = 0.05; b <=1; b+=0.05)
		  {
              for (float c = 0.05; a <=1; a+=0.05)
			  {

                p.x = a;
                p.y = b;
                p.z = c;

                thread zThreads[gNumThreads];

                for(int tid=0; tid < gNumThreads-1; tid++)
				{
                  zThreads[tid] = thread(runIt, p);
                }

                for(int tid=0; tid<gNumThreads-1; tid++)
				{
                  zThreads[tid].join();
                }

              }
          }
      }
  }
  else
  {
      for(int tid=0; tid<8; tid++)
	  {
        runIt(p);
      }
  }

time(&theEnd);
if(MULTITHREAD)
  printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
else
  printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);
  printf(" %d cores computer has\n", number_of_processors);
return 0;

}

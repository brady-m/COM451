#include<stdio.h>
#include <thread>
#include <math.h>
#include <vector>
#include "attractor.h"
using namespace std;
unsigned gNumThreads = 1;

double sigma = 100;
double rho = 5.0;
double beta = 0.6666;
double t = 0.0015;

std::vector<double> bestCoordinates;
/******************************************************************************/
int runIt(int tid)
{ 
 for (double x = 0; x < 1.0; x+=0.1)
  {
    for (double y = 0; y < 1.0; y+=0.1)
    {
      for (double z = 0; z < 1.0; z+=0.1)
      {
        int lorenzIterationCount = 10000000;
        double x1=0;
        double y1=0;
        double z1=0;
        for (int i = 0; i < lorenzIterationCount; i++ )
        {
          double xt = t * (-(y + z));
          double yt = t * x + sigma * y;
          double zt = t * rho + x * z - beta * z;

          x1 += xt;
          y1 += yt;
          z1 += zt;
        //printf("[ %f : %f : %f]\n",x,y,z);
        }
        if(isfinite(x1) && isfinite(y1) && isfinite(z1))
        {
          //Uncomment if you want to see the good coordinates
          //printf("Good coordinate are x= %f, y=%f, z= %f \n",x,y,z);
          bestCoordinates.push_back(x);
	  bestCoordinates.push_back(y);
	  bestCoordinates.push_back(z);
        }
      }      
    }    
  }  
  return 0;
}

/******************************************************************************/
std::vector<double> calculateAttractor(int multithread)
{
  int MULTITHREAD = 0; // set default
  if(multithread != 0)
  {
    MULTITHREAD = multithread;
  }

  time_t theStart, theEnd;
  time(&theStart);
  gNumThreads = std::thread::hardware_concurrency();
  if(MULTITHREAD)
  {
    printf("Amount of threads: %d\n", gNumThreads);
    thread zThreads[gNumThreads];
    for(int tid=0; tid < gNumThreads-1; tid++)
    {
      zThreads[tid] = thread(runIt, tid);
    }

    runIt(gNumThreads-1);
    for(int tid=0; tid<gNumThreads-1; tid++)
    {
      zThreads[tid].join();
  //    zThreads[tid].detach(); // OR USE THIS TO DETACH THE THREAD(S)
    }
  }
  else
  {
      for(int tid=0; tid<8; tid++)
      {
        runIt(tid);
      }
  }

  time(&theEnd);
  if(MULTITHREAD)
    printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
  else
    printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);   
  
  return bestCoordinates;
}

/******************************************************************************/

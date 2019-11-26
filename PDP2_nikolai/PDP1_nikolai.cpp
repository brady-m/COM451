#include <stdio.h>
#include <thread>
#include <math.h>

/* Nikolai Kolomeitse ID7008*/

using namespace std;
int gNumThreads = 8;

float x = 0.5;
float y = 0.5;
float z = 0.5;

/******************************************************************************/

double drand ( double low, double high )
{
    return ( (double)rand() * ( high - low ) ) / (double)RAND_MAX + low;
}

/******************************************************************************/

int runIt(int tid){
  float t = .01; // time step size

  float sigma = -100.0;
  float rho = 1.0;
  float beta = 0.3;

  for (double i = 0; i < 20; ++i)
  {
    x = drand(0.1, 1.0);
    for (double j = 0; j < 20; j++)
    {
      y = drand(0.1, 1.0);
      for (double k = 0; k < 20; k++)
      {
        z = drand(0.1, 1.0);

        for(int i=0; i<10000000; i++){
          float x_n = t * sigma * cos(y);
          float y_n = t * (rho + x - z);
          float z_n = t * beta * (y - z);

          x += x_n;
          y += y_n;
          z += z_n;
    // Uncomment to print values of x y z
    //      printf("tid: %d running iter %d! x=%f y =%f z=%f  \n", tid, i, x,y,z);
        }
      }
    }
  }
  return 0;
}

/******************************************************************************/
int runAss1(int number){

  unsigned numberOfCores = std::thread::hardware_concurrency();
  gNumThreads = numberOfCores;
  printf("The number of cores : %d\n", gNumThreads);

  int MULTITHREAD = 1; // set default
  if(number == 2){
    MULTITHREAD = number;
  }

  time_t theStart, theEnd;
  time(&theStart);


  if(MULTITHREAD){
    thread zThreads[gNumThreads];
    for(int tid=0; tid < gNumThreads-1; tid++){
      zThreads[tid] = thread(runIt, tid);
    }

    runIt(gNumThreads-1);
    for(int tid=0; tid<gNumThreads-1; tid++){
      zThreads[tid].join();
  //    zThreads[tid].detach(); // OR USE THIS TO DETACH THE THREAD(S)
    }
  }

  else{
      for(int tid=0; tid<8; tid++){
        runIt(tid);
        }
  }

  time(&theEnd);
  if(MULTITHREAD)
    printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
  else
    printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);
  return 0;
}

/******************************************************************************/

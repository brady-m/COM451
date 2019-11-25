#include <thread>
#include <iostream>

using namespace std;

double randomCoordinateGenerator(double firstValue, double lastValue);
// int runIt (int tid);

// unsigned int nthreads = 20;
int randomInitialCoordinates = 8;
unsigned gNumThreads = thread::hardware_concurrency();
int numOfUpdates = 10000000;
double dt = 0.1;
double a = 0.1;
double b = 0.1;
double c = 14.0;

double randomCoordinateGenerator(double firstValue, double lastValue){

    return firstValue + (rand() / (RAND_MAX / (firstValue-lastValue)));
}

  int runIt (int tid){

    for(int i = 0; i < randomInitialCoordinates; i++){

      double x, y, z;

      x = randomCoordinateGenerator(0.1, 1);
      y = randomCoordinateGenerator(0.1, 1);
      z = randomCoordinateGenerator(0.1, 1);

      //rossler's attractor
      double xt = (-y - z) * dt;
      double yt = (x + a * y) * dt;
      double zt = (b + z  * (x - c)) * dt;

    for(int i = 0; i < numOfUpdates; i++){

      x += xt;
      y += yt;
      z += zt;

      printf("[ %f : %f : %f]\n", x, y, z);

    }
  }
  return 0;
}


int main(int argc, char* argv[])
{
  cout << gNumThreads << " concurrent threads are supported\n";

  int MULTITHREAD = 1;
  if(argc == 2)
  {
    MULTITHREAD = atof(argv[1]);
  }

  time_t theStart, theEnd;
  time(&theStart);


  if(MULTITHREAD)
  {
    thread zThreads[gNumThreads];
    for(int tid=0; tid < gNumThreads-1; tid++)
    {
      zThreads[tid] = thread(runIt, tid);
    }

    runIt(gNumThreads-1);
    for(int tid=0; tid<gNumThreads-1; tid++)
    {
      zThreads[tid].join();
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

  return 0;
}

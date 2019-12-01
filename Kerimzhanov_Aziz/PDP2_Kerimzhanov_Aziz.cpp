
#include <stdio.h>
#include <thread>

using namespace std;

// these should be a, b, c, defined 8,000 differnt combinations
//double o = 7;
//double p = 28.0;
//double beta = 2.6666;


double t = 0.015;


int runIt(int a, int b, int c, int* val)
{
    *val = 1;

    int max = 1;
    int min = -1;
    float x = rand() % max + min;
    float y = rand() % max + min;
    float z = rand() % max + min;

    float dx, dy, dz;
    for (int i = 0; i < 10000000; i++ )
    {
        dx = x + t * c * (y - x);
        dy = y + t * (x * (a - z) - y);
        dz = z + t * ((x * y) - (b * z));
        x += dx;
        y += dy;
        z += dz;
    }
    if (dx + dy + dz < 0.001) {
      *val = 0;
    }
    if (x + y + z > 50) {
      *val = 0;
    }
}

bool isVal(int vals[], int n) {
  bool isAll = true;
  for (int i = 0;i < n;i++) {
    if (vals[i] == 0) {
      isAll = false;
      break;
    }
  }
  return isAll;
}

int assign()
{
  int gNumThreads = 8;
  int MULTITHREAD = 0; // set default
  unsigned numCPU = thread::hardware_concurrency();
    printf("Available CPU cores: %d\n", numCPU);


  time_t theStart, theEnd;
  time(&theStart);

  int vals[gNumThreads];
  thread zThreads[gNumThreads];
  // for a,b,c parameters (8,000)
  for(float a = 0.05;a < 1;a += 0.05) { // 20 x
    for(float b = 0.05;b < 1;b += 0.05) { // 20 x
      for(float c = 0.05;c < 1;c += 0.05) { // 20 = 8,000
          for(int tid=0; tid < gNumThreads; tid++)
          {
            zThreads[tid] = thread(runIt, a, b, c, &vals[tid]);
          }
          for(int tid=0; tid<gNumThreads; tid++)
          {
            zThreads[tid].join();
          }
          if (isVal(vals, gNumThreads)) {
            printf("Good parameters: %.2f, %.2f, %.2f\n", a, b, c);
          } else {
            printf("parameters are invalid\n");
          }
      }
    }
  }
  time(&theEnd);
  printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
  free(zThreads);
  return 0;
}

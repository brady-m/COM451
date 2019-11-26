

#include <stdio.h>
#include <cstdlib>  // includes atoi() and atof()
#include <string.h> // used by crack.h
#include "crack.h"
#include <thread>
#include "gpu_main.h"
#include "interface.h"
#include "PDP1_Pak.h"
#include "PDP2_Pak.h"


using namespace std;

int visualizeAss1();


int main(int argc, char *argv[]){

  unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
  int gNumThreads = concurentThreadsSupported;

  
  unsigned char ch;
  AParams PARAMS;

  setDefaults(&PARAMS);

  while((ch = crack(argc, argv, "r|v|a|b|", 0)) != 0) {
  	switch(ch){
      case 'r' : PARAMS.runMode = atoi(arg_option); break;
      case 'v' : PARAMS.verbose = 1; break;
      case 'a' : PARAMS.myParam1 = atoi(arg_option); break;
      case 'b' : PARAMS.myParam2 = atof(arg_option); break;
      default  : usage(); return(0);
    }
  }

  if (PARAMS.verbose) viewParams(&PARAMS);

  switch(PARAMS.runMode){
      case 0:
          if (PARAMS.verbose) printf("\n -- running in runMode = 0 -- \n");
          printf("\n -- MAIN INFO -- \n");

          printf("The number of cores : %d\n", gNumThreads);

          int nDevices;

          cudaGetDeviceCount(&nDevices);


          cudaDeviceProp prop;
          cudaGetDeviceProperties(&prop, 0);
          printf("Device name: %s\n", prop.name);
          printf("Total global memory: %Iu\n", prop.totalGlobalMem);
          printf("Maximum amount of shared memory per block: %Iu\n", prop.sharedMemPerBlock);
          printf("Maximum amount of threads per block: %i\n", prop.maxThreadsPerBlock);
          printf("Maximum number of block per dimension of the grid: %i\n", prop.maxGridSize);
          printf("Amount of available constant memory: %Iu\n", prop.totalConstMem);
          printf("Number of multiprocessros on the GPU card: %i\n", prop.multiProcessorCount);
          printf("Number of CPU cores on the machine: %i\n", thread::hardware_concurrency());

          break;

      case 1:
          if (PARAMS.verbose) printf("\n -- running in runMode = 1 -- \n");
          printf("\n RUN ASSIGNMENT 1 \n");

          runAss1(PARAMS.myParam1);

          break;

      case 2:
          if (PARAMS.verbose) printf("\n -- running in runMode = 2 -- \n");
          printf("\n ANIMATE ATTRACTOR \n");

          animateAttractor();

          break;        

      default: printf("no valid run mode selected\n");
  }

return 0;
}

int setDefaults(AParams *PARAMS){

    PARAMS->verbose     = 0;
    PARAMS->runMode     = 1;
    PARAMS->myParam1    = 42;
    PARAMS->myParam2    = 3.14;

    return 0;
}

int usage()
{
  printf("\nUSAGE:\n");
  printf("-r[int] -v -a[int] -b[float] \n\n");
  printf("e.g.> a.out -r1 -v -a25 -b35.2 \n");
  printf("v  verbose mode\n");
  printf("r  run mode (1:myFunction1, 2:MyFunction2)\n");
  printf("a  myParam1 (int)\n");
  printf("b  myParam1 (float)\n");
  printf("\n");
  return(0);
}

int viewParams(const AParams *PARAMS){

  printf("\n--- USING PARAMETERS: ---\n");
  printf("run mode: %d\n", PARAMS->runMode);
  printf("verbose: %d\n", PARAMS->verbose);
  printf("myParam1: %d\n", PARAMS->myParam1);
  printf("myParam2: %f\n", PARAMS->myParam2);
  printf("\n");
  return 0;
}


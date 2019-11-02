/*******************************************************************************
*
*   An example program for how to use use crack.h to collect params and args
*   and file names and so on from the command line
*
*   compile with:> g++ -w interface.cpp
*   run with something like e.g. : a.out -r2 -v -a12 -b5.555
*
*******************************************************************************/
#include <stdio.h>
#include <cstdlib>  // includes atoi() and atof()
#include <string.h> // used by crack.h
#include "crack.h"
#include "interface.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include "attractor.h"
#include "visualizeAttractor.h"
/******************************************************************************/
int main(int argc, char *argv[]){

  unsigned char ch;
  AParams PARAMS;

  setDefaults(&PARAMS);
  std::vector<double> bestCoordinates;
  // -- get parameters that differ from defaults from command line:
  while((ch = crack(argc, argv, "r|v|a|b|", 0)) != 0) {
  	switch(ch){
      case 'r' : PARAMS.runMode = atoi(arg_option); break;
      case 'v' : PARAMS.verbose = 1; break;
      case 'a' : PARAMS.myParam1 = atoi(arg_option); break;
      case 'b' : PARAMS.myParam2 = atof(arg_option); break;
      default  : usage(); return(0);
    }
  }

  // if running in verbose mode, print parameters to screen
  if (PARAMS.verbose) viewParams(&PARAMS);

  // run the system depending on runMode
  switch(PARAMS.runMode){
      case 0:
          if (PARAMS.verbose) printf("\n -- RunMode = 0. Information about PC. \n");
          ShowDeviceInformation();
          break;
      case 1:
          if (PARAMS.verbose) printf("\n -- RunMode = 1. Calculate attractor -- \n");
          // Param1 here defines runinng in multhread mode or not 
          bestCoordinates = calculateAttractor(PARAMS.myParam1);
          break;

      case 2:
          if (PARAMS.verbose) printf("\n -- RunMode = 2. Visualize attractor -- \n");
          visualizeAttractor(bestCoordinates);
          break;

      case 3:
          // and so on...
          break;

      default: printf("no valid run mode selected\n");
  }

return 0;
}


/*******************************************************************************
                       INTERFACE HELPER FUNCTIONS
*******************************************************************************/
int setDefaults(AParams *PARAMS){

    PARAMS->verbose     = 0;
    PARAMS->runMode     = 1;
    PARAMS->myParam1    = 42;
    PARAMS->myParam2    = 3.14;

    return 0;
}

/******************************************************************************/
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

/******************************************************************************/
int viewParams(const AParams *PARAMS){

  printf("\n--- USING PARAMETERS: ---\n");
  printf("run mode: %d\n", PARAMS->runMode);
  printf("verbose: %d\n", PARAMS->verbose);
  printf("myParam1: %d\n", PARAMS->myParam1);
  printf("myParam2: %f\n", PARAMS->myParam2);
  printf("\n");
  return 0;
}
void ShowDeviceInformation()
{
   cudaDeviceProp  prop;
   int count;    
   unsigned numCores = std::thread::hardware_concurrency();
   cudaGetDeviceCount( &count );
   for (int i=0; i< count; i++) {
      cudaGetDeviceProperties( &prop, i );
      printf( "   --- Information for device %d ---\n", i );
      printf( "Name:  %s\n", prop.name );
      printf( "Max amount of shared memory per block:  %ld\n", prop.sharedMemPerBlock );              
      printf( "Max number of threads per block:  %d\n", prop.maxThreadsPerBlock );
      printf( "Max number of blocks per dimension of the grid:  %d\n", prop.maxGridSize );       
      printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
      printf( "Mumber of multiprocessors on the GPU card:  %d\n", prop.multiProcessorCount );
  }
  printf( "Number of cores: %d\n", numCores);
}
/******************************************************************************/

#include <stdio.h>
#include <cstdlib> 
#include <string.h> 
#include "crack.h"
#include "interface.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "attractor.h"
#include "visualizeAttractor.h"

void ShowDeviceInformation() {
   cudaDeviceProp  prop;
   int numberOfDevices;    
   unsigned numberOfCores = std::thread::hardware_concurrency();
   printf( "Number of cores: %d\n", numberOfCores);
   cudaGetDeviceCount(&numberOfDevices);
   for (int i = 0; i < numberOfDevices; i++) {
      cudaGetDeviceProperties( &prop, i );
      printf( "Name: %s\n", prop.name );
      printf( "Max amount of shared memory per block: %ld\n", prop.sharedMemPerBlock );              
      printf( "Max number of threads per block: %d\n", prop.maxThreadsPerBlock );
      printf( "Max number of blocks per dimension of the grid: %d\n", prop.maxGridSize );       
      printf( "Total constant memory: %ld\n", prop.totalConstMem );
      printf( "Mumber of multiprocessors on the GPU card: %d\n", prop.multiProcessorCount );
  }
}



int main(int argc, char *argv[]){

  unsigned char ch;
  AParams PARAMS;

  setDefaults(&PARAMS);

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
  printf("\nUSAGE:\n");
  printf("-r[int] -v -a[int] -b[float] \n\n");
  printf("e.g.> a.out -r1 -v -a25 -b35.2 \n");
  printf("v  verbose mode\n");
  printf("r  run mode (1:myFunction1, 2:MyFunction2)\n");
  printf("a  myParam1 (int)\n");
  printf("b  myParam1 (float)\n");
  printf("\n");
  printf("To run the first assignment in non-Multhireading mode\n");
  printf("You should specify -a0 parameter. e.g:\n");
  printf("./PDP2_LikhovodKirill -r1 -a0\n");
  printf("\n");
  // run the system depending on runMode
  switch(PARAMS.runMode){
      case 0:
          if (PARAMS.verbose) printf("\n RunMode = 0. Information about PC. \n");
          ShowDeviceInformation();
          break;
      case 1:
          if (PARAMS.verbose) printf("\n -- RunMode = 1. Calculate attractor -- \n");
          // Param1 here defines runinng in multhread mode or not 
          RunAssignment1(PARAMS.myParam1);
          break;
      case 2:
          if (PARAMS.verbose) printf("\n -- RunMode = 2. Visualize attractor -- \n");
          visualizeAttractor();
          break;

      default: printf("no valid run mode selected\n");
  }

  return 0;
}

int setDefaults(AParams *PARAMS) {

    PARAMS->verbose     = 0;
    PARAMS->runMode     = 1;
    PARAMS->myParam1    = 42;
    PARAMS->myParam2    = 3.14;

    return 0;
}

int usage() {
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

int viewParams(const AParams *PARAMS) {

  printf("\n--- USING PARAMETERS: ---\n");
  printf("run mode: %d\n", PARAMS->runMode);
  printf("verbose: %d\n", PARAMS->verbose);
  printf("myParam1: %d\n", PARAMS->myParam1);
  printf("myParam2: %f\n", PARAMS->myParam2);
  printf("\n");
  return 0;
}
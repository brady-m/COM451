#include <stdio.h>
#include <cstdlib>
#include <string.h>

#include "crack.h"
#include "params.h"
#include "palette.h"
#include "interface.h"

int main(int argc, char *argv[])
{
  unsigned char i;
  Parameters params;

  setDefaultsParameters(&params);

  while((i = crack(argc, argv, "r|v|a|b|", 0)) != 0)
  {
  	switch(i)
    {
      case 'r' : params.runMode = atoi(arg_option); break;
      case 'v' : params.verbose = 1; break;
      case 'a' : params.param01 = atoi(arg_option); break;
      case 'b' : params.param01 = atof(arg_option); break;
      default  : usage(); return(0);
    }
  }

  if (params.runMode == 0)
  {
    if (params.verbose)
      printComputerInformation();
  }
  else if (params.runMode == 1)
  {
    if (params.verbose)
      attractor();
  }
  else if (params.runMode == 2)
  {
    if (params.verbose)
      drawAnimationPalette();
  }
  else
  {
    printf("incorrect command\n");
  }
}

int usage()
{
  printf("\nUSAGE:\n");
  printf("-r[int] -v -a[int] -b[float] \n\n");
  printf("e.g.> a.out -r1 -v -a25 -b35.2 \n");
  printf("v  verbose mode\n");
  printf("r  run mode (1:myFunction1, 2:MyFunction2)\n");
  printf("a  param01 (int)\n");
  printf("b  param02 (float)\n");
  printf("\n");

  return(0);
}

int setDefaultsParameters(Parameters *p)
{
  p->verbose = 0;
  p->runMode = 1;
  p->param01 = 42;
  p->param01 = 3.14;

  return 0;
}

int viewParameters(const Parameters *p)
{
  printf("\n--- USING PARAMETERS: ---\n");
  printf("run mode: %d\n", p->runMode);
  printf("verbose: %d\n", p->verbose);
  printf("param01: %d\n", p->param01);
  printf("param02: %f\n", p->param02);
  printf("\n");
  return 0;
}

void printComputerInformation()
{
  cudaDeviceProp prop;
  int numOfDevices;

  cudaGetDeviceCount(&numOfDevices);

  for (int i=0; i < numOfDevices; i++)
  {
    cudaGetDeviceProperties(&prop, i);
  	printf("GPU card #%d: %s\n", i, prop.name);
  	printf("Total Global Memory of the GPU card: %ld bytes\n", prop.totalGlobalMem);
  	printf("Max amount of shared memory per block:%ld bytes\n", prop.sharedMemPerBlock);
  	printf("Max number of threads per block: %d\n", prop.maxThreadsPerBlock);
  	printf("Maximum number of blocks 1st dimension of the grid: %d\n", prop.maxThreadsDim[0]);
  	printf("Maximum number of blocks 2nd dimension of the grid: %d\n", prop.maxThreadsDim[1]);
  	printf("Maximum number of blocks 3rd dimension of the grid: %d\n", prop.maxThreadsDim[2]);
  	printf("Total constant memory: %ld\n", prop.totalConstMem);
  	printf("Number of multiprocessors on the GPU card: %d\n", prop.multiProcessorCount);
  }
}

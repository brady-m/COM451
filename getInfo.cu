/*******************************************************************************
*
*   probe a comuter for basic info about processing cores and GPU
*			program by Mike Brady
*
*		/usr/local/cuda/bin/nvcc getInfo.cu -L/usr/local/cuda-10.1/lib64 -I/usr/local/cuda-10.1/targets/x86_64-linux/include -lcuda -lcudart
*
*******************************************************************************/
#include <stdio.h>
//#include <thread>
//#include <sysinfo.h>
#include <unistd.h>

/******************************************************************************/
int main(int argc, char *argv[]){

  cudaError_t err;
  err = cudaDeviceReset();

	printf("************ Machine Info ************\n\n");
	// can also use lscpu
//  unsigned int numCores = std::thread::hardware_concurrency();
	int numCores = sysconf( _SC_NPROCESSORS_ONLN );
	printf("number of usable CPU cores: %d \n", numCores);
	printf("try: \"lscpu\" from command line to get more CPU info\n\n");



  // GET INFORMATION ABOUT GPU DEVICE(S)
  cudaDeviceProp prop;
  int count;
  err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    printf("problem getting device count = %s\n", cudaGetErrorString(err));
    return 1;
    }
  printf("number of GPU devices: %d\n\n", count);

  for (int i = 0; i< count; i++){
    printf("************ GPU Device: %d ************\n\n", i);
    err = cudaGetDeviceProperties(&prop, i);
    if(err != cudaSuccess){
      printf("problem getting device properties = %s\n", cudaGetErrorString(err));
      return 1;
      }

    printf("\tName: %s\n", prop.name);
    printf( "\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf( "\tClock rate: %d\n", prop.clockRate );
    printf( "\tDevice copy overlap: " );
      if (prop.deviceOverlap)
        printf( "Enabled\n" );
      else
        printf( "Disabled\n" );
    printf( "\tKernel execition timeout: " );
      if (prop.kernelExecTimeoutEnabled)
        printf( "Enabled\n" );
      else
        printf( "Disabled\n" );
//    printf( "--- Memory Information for device %d ---\n", i );
    printf("\tTotal global mem: %ld\n", prop.totalGlobalMem );
    printf("\tTotal constant Mem: %ld\n", prop.totalConstMem );
    printf("\tMax mem pitch: %ld\n", prop.memPitch );
    printf( "\tTexture Alignment: %ld\n", prop.textureAlignment );
    printf("\n");
    printf( "\tMultiprocessor count: %d\n", prop.multiProcessorCount );
    printf( "\tShared mem per processor: %ld\n", prop.sharedMemPerBlock );
    printf( "\tRegisters per processor: %d\n", prop.regsPerBlock );
    printf( "\tThreads in warp: %d\n", prop.warpSize );
    printf( "\tMax threads per block: %d\n", prop.maxThreadsPerBlock );
    printf( "\tMax block dimensions: (%d, %d, %d)\n",
                  prop.maxThreadsDim[0],
                  prop.maxThreadsDim[1],
                  prop.maxThreadsDim[2]);
    printf( "\tMax grid dimensions: (%d, %d, %d)\n",
                  prop.maxGridSize[0],
                  prop.maxGridSize[1],
                  prop.maxGridSize[2]);
    printf("\n");
  }

  return 0;
}

#include <iostream>
#include <stdio.h>
#include <sys/sysinfo.h>
#include "errorHandler.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main( void ) {

    cudaDeviceProp  prop;
    int count;

    HANDLE_ERROR( cudaGetDeviceCount( &count ) );

    for (int i = 0; i < count; i++) {

        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total Global Memory of the GPU card:  %ld\n", prop.totalGlobalMem);
        printf( "Amount of available constant memory:  %ld\n", prop.totalConstMem);
        printf( "Maximum amount of shared memory per block:  %ld\n", prop.sharedMemPerBlock);
        printf( "Maximum number of threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf( "The number of blocks allowed along each dimension of a grid:  %d\n", prop. maxGridSize[3]);
        printf( "The number of multiprocessors on the device:  %d\n", prop.multiProcessorCount);
        printf("Number of CPU cores on the machine %d\n", get_nprocs_conf());

      }
    }

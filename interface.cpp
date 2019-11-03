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
#include <iostream>
#include "PDP1_ershov.h"
#include "visualization.h"
#include "interface.h"


/******************************************************************************/
int main(int argc, char *argv[]){

	unsigned char ch;
	AParams PARAMS;

	setDefaults(&PARAMS);

	// -- get parameters that differ from defaults from command line:
	while((ch = crack(argc, argv, "r|v|a|b|", 0)) != 0) {

		switch(ch){
			case 'r' : PARAMS.runMode = atoi(arg_option); break;
			case 'v' : PARAMS.verbose = 1; break;
			case 'a' : PARAMS.isMultithread = (bool) atoi(arg_option); break;
			//case 'b' : PARAMS.isMultithread = (bool) atoi(arg_option); break;
			default  : usage(); return(0);
		}
	}

	// if running in verbose mode, print parameters to screen
	if (PARAMS.verbose) viewParams(&PARAMS);

	// run the system depending on runMode
	switch(PARAMS.runMode){
			case 0:
					if (PARAMS.verbose) printf("\n -- running information about the GPU  -- \n");
					showDeviceInformation();
					break;

			case 1:
					if (PARAMS.verbose) printf("\n -- running attractor calculation -- \n");
					calcAttractor(PARAMS.isMultithread);
					break;

			case 2:
					if (PARAMS.verbose) printf("\n -- running attractor visualization -- \n");
					drawAttractor();
					break;

			default: printf("no valid run mode selected\n");
	}

	return 0;
}


/*******************************************************************************
											 INTERFACE HELPER FUNCTIONS
*******************************************************************************/
int setDefaults(AParams *PARAMS){

		PARAMS->verbose       = 0;
		PARAMS->runMode       = 1;
		//PARAMS->myParam1      = 42;
		PARAMS->isMultithread = true;

		return 0;
}

/******************************************************************************/
int usage()
{
	printf("\nUSAGE:\n");
	printf("-r[int] -v -a[int] \n\n");
	printf("e.g.> a.out -r1 -v -a0 \n");
	printf("v  verbose mode\n");
	printf("r  run mode (1:calculate attractor, 2:draw attractor)\n");
	printf("a  isMultithread (int)\n");
	//printf("b  isMultithread (int)\n");
	printf("\n");
	return(0);
}

/******************************************************************************/
int viewParams(const AParams *PARAMS){

	printf("\n--- USING PARAMETERS: ---\n");
	printf("run mode: %d\n", PARAMS->runMode);
	printf("verbose: %d\n", PARAMS->verbose);
	//printf("myParam1: %d\n", PARAMS->myParam1);
	printf("isMultithread: %d\n", PARAMS->isMultithread);
	printf("\n");
	return 0;
}

/******************************************************************************/

void showDeviceInformation() {
	 cudaDeviceProp prop;
	 int numOfDevices;    
	 cudaGetDeviceCount(&numOfDevices);
	 for (int i=0; i < numOfDevices; i++){
			cudaGetDeviceProperties(&prop, i);
			printf("GPU card #%d: %s\n", i, prop.name);
			printf("Total Global Memory of the GPU card:\t\t\t%ld bytes\n", prop.totalGlobalMem);
			printf("Max amount of shared memory per block:\t\t\t%ld bytes\n", prop.sharedMemPerBlock);              
			printf("Max number of threads per block:\t\t\t%d\n", prop.maxThreadsPerBlock);
			printf("Maximum number of blocks 1st dimension of the grid:\t%d\n", prop.maxThreadsDim[0]);
			printf("Maximum number of blocks 2nd dimension of the grid:\t%d\n", prop.maxThreadsDim[1]);
			printf("Maximum number of blocks 3rd dimension of the grid:\t%d\n", prop.maxThreadsDim[2]);       
			printf("Total constant memory:\t\t\t\t\t%ld\n", prop.totalConstMem);
			printf("Number of multiprocessors on the GPU card:\t\t%d\n", prop.multiProcessorCount);
	}
}
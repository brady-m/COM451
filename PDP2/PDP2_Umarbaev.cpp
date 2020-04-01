/*******************************************************************************
*
*   An example program for how to use use crack.h to collect Parameters and args
*   and file names and so on from the command line
*
*   compile with: make
*   run with: 1) ./PDP2_Umarbaev -r0 -v
*             2) ./PDP2_Umarbaev -r1 -v
*             3) ./PDP2_Umarbaev -r2 -v
*
*******************************************************************************/
#include <stdio.h>
#include "interface.h"
#include "crack.h"
#include "cudaDeviceProp.h"

/******************************************************************************/
int main(int argc, char *argv[]){

  unsigned char ch;
  Parameters Parameters;
  Points Points;
  srand((unsigned)time(NULL));

  setDefaultParams(Parameters);

  // -- get parameters that differ from defaults from command line:
  while((ch = crack(argc, argv, "r|v|a|b|c|d", 0)) != 0) {
  	switch(ch){
      case 'r' : Parameters.runMode = atoi(arg_option); break;
      case 'v' : Parameters.verbose = 1; break;
      case 'a' : Parameters.a = atof(arg_option); break;
      case 'b' : Parameters.b = atof(arg_option); break;
      case 'c' : Parameters.c = atof(arg_option); break;
      default  : usage(); return(0);
    }
  }

  // if running in verbose mode, print parameters to screen
  if (Parameters.verbose) viewParameters(Parameters);

  // run the system depending on runMode
  switch(Parameters.runMode){

      case 0:
          if (Parameters.verbose) printf("\n -- running mode 0 function -- \n");
          cudaDeviceProp  prop;
          printCudaDeviceProperties(prop);
          break;

      case 1:
          if (Parameters.verbose) printf("\n -- running mode 1 function -- \n");
          runMode1(Parameters);
          break;

      case 2:
          if (Parameters.verbose) printf("\n -- running mode 2 function -- \n");
          runMode2(Parameters, Points);
          break;

      default: printf("no valid run mode selected\n");
  }

  return 0;
}

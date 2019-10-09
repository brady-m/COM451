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

// --- TODO: create interface.h library, move these there, and crack to here
// --- also, update crack function to not need the -w flag at compile time
struct AParams {
  bool  verbose;
  int   runMode;
  int   myParam1;
  float myParam2;
};
int usage();
int setDefaults(AParams *PARAMS);
int viewParams(const AParams *PARAMS);
// ---

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
      case 'a' : PARAMS.myParam1 = atoi(arg_option); break;
      case 'b' : PARAMS.myParam2 = atof(arg_option); break;
      default  : usage(); return(0);
    }
  }

  // if running in verbose mode, print parameters to screen
  if (PARAMS.verbose) viewParams(&PARAMS);

  // run the system depending on runMode
  switch(PARAMS.runMode){
      case 1:
          if (PARAMS.verbose) printf("\n -- running in runMode = 1 -- \n");
          // insert function of method for runMode 1 here, for example:
          // myFunction1(&PARAMS);
          // also change verbose message above to something more descriptive
          // like, " -- running myFunction1 -- "
          break;

      case 2:
          if (PARAMS.verbose) printf("\n -- running in runMode = 2 -- \n");
          // insert function of method for runMode 1 here, for example:
          // myFunction2(&PARAMS);
          // also change verbose message above to something more descriptive
          // like, " -- running myFunction2 -- "
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

/******************************************************************************/

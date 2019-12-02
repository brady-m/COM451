#include "struct.h"

#include <string.h>
#include <iostream>
#include <stdlib.h>

int arg_index = 0;
char *arg_option;
char *pvcon = NULL;

char crack(int argc, char** argv, char* flags, int ignore_unknowns)
{
    char *pv, *flgp;

    while ((arg_index) < argc){
        if (pvcon != NULL)
            pv = pvcon;
        else{
            if (++arg_index >= argc) return(0);
            pv = argv[arg_index];
            if (*pv != '-')
                return(0);
            }
        pv++;

        if (*pv != 0){
            if ((flgp=strchr(flags,*pv)) != NULL){
                pvcon = pv;
                if (*(flgp+1) == '|') { arg_option = pv+1; pvcon = NULL; }
                return(*pv);
                }
            else
                if (!ignore_unknowns){
                    fprintf(stderr, "%s: no such flag: %s\n", argv[0], pv);
                    return(EOF);
                    }
                else pvcon = NULL;
	    	}
        pvcon = NULL;
    }

    return(0);
}

int setDefaultParams(Parameters& Parameters)
{
  Parameters.verbose     = 0;
  Parameters.runMode     = 2;
  Parameters.a           = 10.0;
  Parameters.b           = 28.0;
  Parameters.c           = 2.666;

  return 0;
}

int usage()
{
  printf("\nUSAGE:\n");
  printf("-r[int] -v -a[double] -b[double] -c[double] -d[double] \n\n");
  printf("e.g.> PDP2_Umarbaev -r1 -v -a5.5 -b35.2 -c5.7 -d3.2 \n");
  printf("v  verbose mode\n");
  printf("r  run mode (0: Device properties, 1: PDP1, 2: PDP2)\n");
  printf("a  (double)\n");
  printf("b  (double)\n");
  printf("c  (double)\n");
  printf("\n");
  return(0);
}

int viewParameters(const Parameters& Parameters)
{
  printf("\n--- USING PARAMETERS: ---\n");
  printf("run mode: %d\n", Parameters.runMode);
  printf("verbose: %d\n", Parameters.verbose);
  printf("a: %f\n", Parameters.a);
  printf("b: %f\n", Parameters.b);
  printf("c: %f\n", Parameters.c);
  printf("\n");
  return 0;
}

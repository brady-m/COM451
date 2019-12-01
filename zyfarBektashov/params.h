#ifndef hInterfaceLib
#define hInterfaceLib


#include "gpu_main.h"

struct Parameters {
    bool  verbose;
    int   runMode;
    int   param01;
    float param02;
};


int usage();
int setDefaultsParameters(Parameters *p);
int viewParameters(const Parameters *p);
int attractor();
void printComputerInformation();
void drawAnimationPalette();
#endif

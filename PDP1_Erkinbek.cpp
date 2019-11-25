#include <stdio.h>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <math.h>


using namespace std;


int numberThreads = 8;
int iterations = 10000000;

float q = 0;
float p = 0;
float b = 0;
float i = 0;
float j = 0;
float k = 0;
float t = 0.01;

float varX = 0.3;
float varY = 0.2;
float varZ = 0.6;


// create random number between 0 and 1
float floatRand ( float low, float high )
{
    return ( (float)rand() * ( high - low ) ) / (float)RAND_MAX + low;
}

int runIt(int tid, float q, float p, float b){

  varX = floatRand(0.1, 1.0);
  varY = floatRand(0.1, 1.0);
  varZ = floatRand(0.1, 1.0);

  // make 10mln number of iterations
  for(int iter=0; iter < iterations; ++iter){
    
    // First formula
    float x = t * q * cos(varY);
    float y = t * (p + varX - varZ);
    float z = t * b * (varY - varZ);

    // Second formula
    // float x = t * ((varZ * varY) / q);
    // float y = t * (varY - varX + p);
    // float z = t * (b + varY);
    
    varX += x;
    varY += y;
    varZ += z;
    //printf("%f, %f, %f\n", q, p, b);
    //printf("tid: %d, running iter %d!, valX=%f, valY=%f, valZ=%f \n", tid, iter, varX, varY, varZ);
  }

  return 0;
}

/******************************************************************************/
int runAss1(int thrNum){

  time_t theStart, theEnd;
  time(&theStart);


  int MULTITHREAD = 1; // set default
  if(thrNum != 1){
    MULTITHREAD = thrNum;
  }


  if(MULTITHREAD){
    thread zThreads[numberThreads];

    for (float i = 0.05; i < 1; i+=0.05){
      for (float j = 0.05; j < 1; j+=0.05){
        for (float k = 0.05; k < 1; k+=0.05){
              for(int tid=0; tid < numberThreads-1; tid++){
                zThreads[tid] = thread(runIt, tid, i, j, k);
              }

              runIt(numberThreads-1, i, j, k);
              for(int tid=0; tid<numberThreads-1; tid++){
                zThreads[tid].join();
              }
        }
      }
    }
  } 
  else {
      for(int tid=0; tid<8; tid++){
        runIt(tid, i, j, k);
        }
  }

  time(&theEnd);
  if(MULTITHREAD){
    printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
  }   
  else {
    printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);
  }
  return 0;
}

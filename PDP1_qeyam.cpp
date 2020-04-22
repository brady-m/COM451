#include <stdio.h>
#include <thread>
#include <math.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

int gNumThreads = 8;
int N = 10000000;

float t = 0.01;

int init(float a, float b, float c, int* theResult){

  	double dX, dY, dZ;

  	double x = rand() % 100 + 1;
  	double y = rand() % 100 + 1;
  	double z = rand() % 100 + 1;

    *theResult = 1; 

  	for(int i = 0; i < N; i++){

      	dX = t * a * cos(y);
      	dY = t * (b + x - z);
      	dZ = t * (c * (y - z));

      	x += dX;
      	y += dY;
      	z += dZ;

      if (dX + dY + dZ < 0.001){
        theResult = 0;
        break;
      }

      if (x + y + z > 500){
        theResult = 0;
        break;
      }
  	}
  	return 0;
}

/*************************Main Function********************************************/
int main(int argc, char* argv[]){

  	int MULTITHREAD = 1;
  	if(argc == 2){
    	MULTITHREAD = atof(argv[1]);
  	}

  	time_t theStart, theEnd;
  	time(&theStart);

    int theResult[gNumThreads];

    if(MULTITHREAD){
        for (float a = 0.05; a <= 1; a += 0.05){
            for (float b = 0.05; b <= 1; b += 0.05){
                for (float c = 0.05; c <= 1; c += 0.05){

                  	thread zThreads[gNumThreads];
                
                  	for(int tid=0; tid < gNumThreads-1; tid++){
                    	zThreads[tid] = thread(init, a, b, c, &theResult[tid]);
                  	}

                  	for(int tid=0; tid<gNumThreads-1; tid++){
                    	zThreads[tid].join();
                  	}

                    // test here if all theResult[] are ones
                }
            }
        }
    }
 	/*else{
    	for(int tid = 0; tid < 8; tid++){
        	sigma -= 0.1;
        	riho -= 0.1;
        	beta -= 0.1;
        	init(tid, sigma, riho, beta);
        }
  	}*/

  	time(&theEnd);

  	if(MULTITHREAD){
    	printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
    }
    else{
    	printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);
    }
  return 0;
}
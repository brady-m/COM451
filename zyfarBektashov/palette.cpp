#include <stdio.h>
#include <thread>
#include <stdlib.h>
#include <math.h>


using namespace std;
int gNumThreads = 8;

struct APoint
{
    float x;
    float y;
    float z;
};



int runIt(APoint p){

	APoint thePoint, theChange;
	float t = 0.005;

    thePoint.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    thePoint.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    thePoint.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

	float sigma = p.x;
	float rho = p.y;
	float beta = p.z;
	int xIdx;
	int yIdx;

	for (long i = 1; i < 10000000; i++)
	{

        int db = thePoint.x + thePoint.y + thePoint.z;
		theChange.x = t * sigma * cos(thePoint.y);
        theChange.y = t * (rho + thePoint.x - thePoint.z);
        theChange.z = t * beta * (thePoint.y - thePoint.z);


		thePoint.x += theChange.x;
		thePoint.y += theChange.y;
		thePoint.z += theChange.z;

        int da = thePoint.x + thePoint.y + thePoint.z;
        if (db != da && thePoint.x < 1080 && thePoint.y < 600) {
            printf("[%f %f %f]\n", thePoint.x, thePoint.y, thePoint.z);
        }

	}

return 0;
}

/******************************************************************************/
int attractor(){

    int MULTITHREAD = 1; // set default

    time_t theStart, theEnd;
    time(&theStart);
    APoint p;

    if(MULTITHREAD){
        for (float a = 0.05; a <=1; a+=0.05){
            for (float b = 0.05; b <=1; b+=0.05){
                for (float c = 0.05; a <=1; a+=0.05){

                p.x = a;
                p.y = b;
                p.z = c;

                thread zThreads[gNumThreads];

                for(int tid=0; tid < gNumThreads-1; tid++){
                    zThreads[tid] = thread(runIt, p);
                }

                for(int tid=0; tid<gNumThreads-1; tid++){
	              zThreads[tid].join();
        	    }

                }
            }
        }
    }


  time(&theEnd);
  if(MULTITHREAD)
    printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
  else
    printf("NOT THREADING seconds used: %ld\n", theEnd - theStart);
  return 0;
}

#include <stdlib.h>
#include <thread>
#include <sched.h>
#include <stdio.h>
#include <sys/sysinfo.h>

/*

	Assingment #1 
	Pak Vladsilav 

*/

using namespace std;
int gNumThreads = 8;

float varO = 0.2;
float varB = 0.2;
float varP = 5.7;
float varT = 0.001; 
float varXvarT;
float varYvarT;
float varZvarT;

/******************************************************************************/
int runIt(int tid){

  float varX = rand() % 20 + 1;
  float varY = rand() % 20 + 1; 
  float varZ = rand() % 20 + 1; 

  int ten_million_operations = 10000000;

  /*  
      I used my own equestion ->

      fx = -(y + z) * t
      fy = t * (x + O * y)
      fz = (B + x * z - P * z) * t
    */
      

  for(int i = 0; i < ten_million_operations; i++){
  	
   
    for(int i = 0; i < 8000; i++) {
      varXvarT = -(varY + varZ) * varT;
      varX += varXvarT;
    }

    for(int i = 0; i < 8000; i++) {
      varYvarT = varT * (varX + varO * varY);
       varY += varYvarT;
    }

    for(int i = 0; i < 8000; i++) {
      varZvarT = (varB + varX  * varZ - varP * varZ) * varT;
        varZ += varZvarT;
    }

   	// float varXvarT = -(varY + varZ) * varT;

    // float varYvarT = varT * (varX + varO * varY);

    // float varZvarT = (varB + varX  * varZ - varP * varZ) * varT;

    varX += varXvarT;

    varY += varYvarT;

    varZ += varZvarT;

    //----------------------------------------------------------------
    // Uncomment code below to print x: y: z:
    // printf("x: %f       y: %f       z: %f\n", varX, varY, varZ);
  }
  
  return 0;
  
}

/******************************************************************************/
int runAss1(int multithread) {

  int MULTITHREAD = 1; 

  if(multithread == 2) {

    MULTITHREAD = multithread;

  }

  unsigned int numberOfCores = std::thread::hardware_concurrency();
  printf("This machine has %d cores\n", numberOfCores);

  	/*
	
	I found how to set cores on the stackoverflow (link below)

  	https://stackoverflow.com/questions/10490756/how-to-use-sched-getaffinity-and-sched-setaffinity-in-linux-from-c
	
	I am also printing wich core is added one by one, to check that everything is working.

	*/

  cpu_set_t mask;
  CPU_ZERO(&mask); 

  for (int i = 0; i < numberOfCores; i++) {

  	CPU_SET(i, &mask);
  	printf("...Core number %d is added\n", i + 1);

    }

  time_t theStart, theEnd;
  time(&theStart);


  if(MULTITHREAD) {

    thread zThreads[gNumThreads];

    for(int tid=0; tid < gNumThreads-1; tid++) {

      zThreads[tid] = thread(runIt, tid);

    }

    runIt(gNumThreads-1);

    for(int tid=0; tid<gNumThreads-1; tid++) {

      zThreads[tid].join();

    }
  }

  else {

      for(int tid=0; tid<8; tid++) {

        runIt(tid);

        }
  }

  time(&theEnd);

  if(MULTITHREAD) {
  	printf("In MULTITHREADING MODE  %ld seconds needed for calculation! \n", theEnd - theStart);
  } else {
    printf("In NOT THREADING MODE  %ld seconds needed for calculation! \n", theEnd - theStart);
  }
    
  return 0;

}

/******************************************************************************/

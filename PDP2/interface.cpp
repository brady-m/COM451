#include "interface.h"

double getRandNum() {
    return double(std::rand()) / (double(RAND_MAX) + 1.0);
}

/*******************************CRACK*******************************************/
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
  Parameters.runMode     = 1;
  Parameters.a           = 10.0;
  Parameters.b           = 28.0;
  Parameters.c           = 2.666;
  // Parameters.d           = 6;

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
  // printf("d  (double)\n");
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
  // printf("d: %f\n", Parameters.d);
  printf("\n");
  return 0;
}


/******************************RunMode 1*******************************************/
int getThreadNum(const std::string mode)         // function to acquire number of cores on machine 
{                      
  return mode.compare("multi") == 0 ? get_nprocs() : 1;  // based on the chosen mode(multi or nonmulti)
}

void calculateEquation(const std::string mode, Parameters& Parameters) 
{
  Point Point;

  for (Parameters.a = 0.05; Parameters.a <= 1; Parameters.a+=0.05) {
    for (Parameters.b = 0.05; Parameters.b <= 1; Parameters.b+=0.05) {
      for (Parameters.c = 0.05; Parameters.c <= 1; Parameters.c+=0.05) {
                
        for (int i = 0; i < TEST_NUM/getThreadNum(mode); ++i) {
          Point.start_x = double(std::rand()) / (double(RAND_MAX) + 1.0);
          Point.start_y = double(std::rand()) / (double(RAND_MAX) + 1.0);
          Point.start_z = double(std::rand()) / (double(RAND_MAX) + 1.0);

          Point.x = Point.start_x;
          Point.y = Point.start_y;
          Point.z = Point.start_z;

          for (int j = 0; j < ITERATION_NUM; ++j) {
            Point.delta_x = t * (Parameters.a * (Point.y - Point.x));
            Point.delta_y = t * ( (Point.x * (Parameters.b - Point.z)) - Point.y);
            Point.delta_z = t * ( (Point.x * Point.y) - (Parameters.c * Point.z) );
            
            Point.x += Point.delta_x;
            Point.y += Point.delta_y;
            Point.z += Point.delta_z;
          }

          double totalChange = (fabs(Point.delta_x) + 
                                fabs(Point.delta_y) + 
                                fabs(Point.delta_z));
                    
          if (totalChange >= 0.01 &&
                Point.x <= 100 && Point.y <= 100 && Point.z <= 100) {
                    
            printf("\nPARAMETERS: sigma=%.2f, rho=%.2f, beta=%.2f\n", 
                    Parameters.a, Parameters.b, Parameters.c);
          }  else {
            printf("These parameters are invalid, searching for valid parameters\n");
          }
        }
      }
    }
  }
}

int runMode1(Parameters& Parameters) 
{
  std::string mode;
  while (1) {

    printf("Choose mode (multi/nonmulti)\n");
    std::cin >> mode;

    if (mode.compare("exit") == 0) break;

    if (mode.compare("multi") == 0) {									// MULTI MODE

      printf("Number of cores: %d\n", getThreadNum(mode));
      std::thread threads[getThreadNum(mode)];

      for (int tId = 0; tId < getThreadNum(mode)-1; ++tId) {                    
        threads[tId] = std::thread(calculateEquation, mode, std::ref(Parameters));
      }

      calculateEquation(mode, Parameters);

      for (int tId = 0; tId < getThreadNum(mode)-1; ++tId) {
        threads[tId].join();
        // threads[tId].detach();
      }
    }
    else if (mode.compare("nonmulti") == 0) {                           // NONMULTI MODE
      calculateEquation(mode, Parameters);
    }
  }

  return 0;
}


/******************************RunMode 2*******************************************/
GPU_Palette openPalette(int theWidth, int theHeight)
{
  unsigned long theSize = theWidth * theHeight;
  unsigned long memSize = theSize * sizeof(float);

  float* redmap = (float*) malloc(memSize);
  float* greenmap = (float*) malloc(memSize);
  float* bluemap = (float*) malloc(memSize);

  for(int i = 0; i < theSize; i++){
    bluemap[i] 	= .0;
    greenmap[i] = .0;
    redmap[i]   = .0;
  }

  GPU_Palette P1 = initGPUPalette(theWidth, theHeight);

  cudaMemcpy(P1.red, redmap, memSize, cH2D);
  cudaMemcpy(P1.green, greenmap, memSize, cH2D);
  cudaMemcpy(P1.blue, bluemap, memSize, cH2D);

  free(redmap);
  free(greenmap);
  free(bluemap);

  return P1;
}

int drawEquation(GPU_Palette* P1, CPUAnimBitmap* A1, 
           const Parameters& Parameters, Point& Point) {

  srand((unsigned)time(NULL));
  Point.start_x = getRandNum();
  Point.start_y = getRandNum();
  Point.start_z = getRandNum();

  Point.x = Point.start_x;
  Point.y = Point.start_y;
  Point.z = Point.start_z;

  for (long i = 1; i < 1000000; i++) {

    Point.delta_x = t * (Parameters.a * (Point.y - Point.x));
    Point.delta_y = t * ( (Point.x * (Parameters.b - Point.z)) - Point.y);
    Point.delta_z = t * ( (Point.x * Point.y) - (Parameters.c * Point.z) );
    
    Point.x += Point.delta_x;
    Point.y += Point.delta_y;
    Point.z += Point.delta_z;

    static float minX = -50.0;
    static float maxX = 50.0;
    static float minY = -60.0;
    static float maxY = 60.0;

    static float xRange = fabs(maxX - minX);
    static float xScalar = 0.9 * (gWIDTH/xRange);

    static float yRange = fabs(maxY - minY);
    static float yScalar = 0.9 * (gHEIGHT/yRange);

    Point.xIdx = round(xScalar * (Point.x - minX));
    Point.yIdx = round(yScalar * (Point.y - minY));

	  // Point.xIdx = floor((Point.x * 32) + 960); // (X * scalar) + (gWidth/2)
	  // Point.yIdx = floor((Point.y * 18) + 540); // (Y * scalar) + (gHeight/2)

    updatePalette(P1, Point);

  }

  return 0;
}

int drawPal(GPU_Palette* P1, CPUAnimBitmap* A1, Points& Points) {
  for (long i = 1; i < 1000000; i++) {
    A1->drawPalette();
  }

  return 0;
}

int runMode2(const Parameters& Parameters, Points& Points)  // ¯\_(ツ)_/¯
{
  GPU_Palette P1;
  P1 = openPalette(gWIDTH, gHEIGHT); // width, height of palette as args 

  CPUAnimBitmap animation(&P1);
  cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

  std::thread threads[NUMBER_OF_POINTS];

  for (int tId = 0; tId < NUMBER_OF_POINTS-1; ++tId) {                    
    threads[tId] = std::thread(drawEquation, &P1, &animation, std::ref(Parameters), 
                               std::ref(Points.points[tId]));
  }
  drawPal(&P1, &animation, Points);
  // threads[NUMBER_OF_POINTS] = std::thread(drawPal, &P1, &animation, std::ref(Points));
  // drawEquation(&P1, &animation, Parameters, Points.points[NUMBER_OF_POINTS-1]);
  for (int tId = 0; tId < NUMBER_OF_POINTS-1; ++tId) {
    threads[tId].join();
  }

  freeGPUPalette(&P1);

  return 0;
}

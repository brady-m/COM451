#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <pthread.h>
#include <sched.h>
#include <math.h>
#include <iomanip>
 
using namespace std;
 
// I create 20 threads with different parameters and test all 20 parameter configurations from 8 random start points
int threadsNumber = 20;
int iterationCount = 10000000;
int randomInitialLocations = 8;
double t = 0.1;

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void calculatePointsOfFigure (int tid, double a, double b, double c, cpu_set_t *mask) {

    if (mask != NULL) {
        sched_setaffinity(0, sizeof(mask), mask);
    }

    int currentCPU = sched_getcpu()+1;

    double x = fRand(1, 10.0);
    double y = fRand(1, 10.0);
    double z = fRand(1, 10.0);  
       
 
    for (int i = 0; i < iterationCount-1; i++ ) {

        // double xt = t * -(y + z);
        // double yt = t * (x + a * y);
        // double zt = t * (b + x * z - c * z);

        // I kept the first formula here because it mostly works well enough without some strange parametrs
        // I also tried the commented formula, but it requires a bit smaller t to not grow till NaN
        double xt = t * a * cos(y);
        double yt = t * (c + x - z);
        double zt = t * b * (y - z);

        x += xt;
        y += yt;
        z += zt;
    }

        // double xt =x + t * -(y + z);
        // double yt =y + t * (x + a * y);
        // double zt =z + t * (b + x * z - c * z);

        double xt = x + t * a * cos(y);
        double yt = y + t * (c + x - z);
        double zt = z + t * b * (y - z);

    printf("Points configuration #%d, started on core #%d, executed on core #%d\n", tid, currentCPU, sched_getcpu()+1);
    if (xt != x ) {
        printf("----------------------------------------------------\n");   
        printf("The point is still moving with parameters %f, %f, %f\n", a,b,c);
        printf("----------------------------------------------------\n");  

        // uncomment the code below to see actual point locations

        // cout << setprecision(20) <<  "x = " << x << endl;
        // cout << setprecision(20) <<  "y = " << y << endl;
        // cout << setprecision(20) <<  "z = " << z << endl;
        // printf("____________________________________________________\n");  
        // cout << setprecision(20) <<  "x = " << xt << endl;
        // cout << setprecision(20) <<  "y = " << yt << endl;
        // cout << setprecision(20) <<  "z = " << zt << endl;
        // printf("\n"); 


    } else {
        printf("Point is NOT moving with parameters %f, %f, %f\n", a,b,c);

        // uncomment the code below to see actual point locations

        // cout << setprecision(20) <<  "x = " << x << endl;
        // cout << setprecision(20) <<  "y = " << y << endl;
        // cout << setprecision(20) <<  "z = " << z << endl;
        // printf("____________________________________________________\n");  
        // cout << setprecision(20) <<  "x = " << xt << endl;
        // cout << setprecision(20) <<  "y = " << yt << endl;
        // cout << setprecision(20) <<  "z = " << zt << endl;
        // printf("\n"); 
    }
}
 
  
int main(int argc, char* argv[]) {   

    typedef chrono::high_resolution_clock Time;
    typedef chrono::duration<float> fsec;

    unsigned numCPU = thread::hardware_concurrency();
    cout << "Number of cores on this machine is " << numCPU << endl;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset); 
    for (int i = 0; i < numCPU; i++) {
        CPU_SET(i, &cpuset);
    }

    int MULTITHREAD = 1; 
    if(argc == 2){
        MULTITHREAD = atof(argv[1]);
    }

    auto begin = Time::now();

    if (MULTITHREAD) {

        vector<thread> threads; 

        for (int i = 0; i < threadsNumber; i++) {

            double randomA = fRand(0.0, 1.0);
            double randomB = fRand(0.0, 1.0);
            double randomC = fRand(0.0, 1.0);

            for (int j = 0; j < randomInitialLocations; j++) {
                threads.emplace_back(thread(calculatePointsOfFigure, i, randomA, randomB, randomC, &cpuset));
            }
        }
    
        for (auto& th : threads) {
            th.join();
        } 

    } else {

        for (int i = 0; i < threadsNumber; i++) {

            double randomA = fRand(0.0, 1.0);
            double randomB = fRand(0.0, 1.0);
            double randomC = fRand(0.0, 1.0);

            for (int i = 0; i < threadsNumber; i++) {

            double randomA = fRand(0.0, 1.0);
            double randomB = fRand(0.0, 1.0);
            double randomC = fRand(0.0, 1.0);

                for (int j = 0; j < randomInitialLocations; j++) {
                    calculatePointsOfFigure(i, randomA, randomB, randomC, &cpuset);
                }
            }
        }
    }

    auto end = Time::now();
    fsec fs = end - begin;

    if (MULTITHREAD) {
        printf("----------------------------------------------------\n");   
        cout << fs.count() << " seconds used in multithreading mode" << endl;
        printf("----------------------------------------------------\n");   
    } else {
        printf("----------------------------------------------------\n");   
        cout << fs.count() << " seconds used in non-multithreading mode" << endl;
        printf("----------------------------------------------------\n");   
    }
}

#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <pthread.h>
#include <sched.h>
#include <math.h>
 
using namespace std;
 
int threadsNumber = 8;
int iterationCount = 10000000;

double t = 0.1;

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void testAllRandomPoints(double a, double b, double c, vector<int> result) {
    int sum = 0;
    for (int i = 0; i < result.size();i++) {
        sum += result[i];
    }

    if (sum == 8) {
        printf("Parameters %f %f %f satisfy the requirements\n", a, b,c);
    }
}

void calculatePointsOfFigure (double a, double b, double c, int* result) {

    double x = fRand(1, 10.0);
    double y = fRand(1, 10.0);
    double z = fRand(1, 10.0);  

    double xt, yt, zt;
 
    for (int i = 0; i < iterationCount; i++ ) {

        xt = t * a * cos(y);
        yt = t * (c + x - z);
        zt = t * b * (y - z);

        x += xt;
        y += yt;
        z += zt;
    }

    if (xt + yt + zt < 0.001) {
        *result = 0;
        return;
    }

    if (x + y + z > 100) {
        *result = 0;
        return;
    }

    *result = 1; 
}
 
  
int RunAssignment1(int runMode) {   

    typedef chrono::high_resolution_clock Time;
    typedef chrono::duration<float> fsec;

    unsigned numCPU = thread::hardware_concurrency();
    cout << "Number of cores on this machine is " << numCPU << endl;
    threadsNumber = numCPU;

    int MULTITHREAD = 1; 
    if(runMode == 0){
        MULTITHREAD = 0;
    }

    auto begin = Time::now();

    if (MULTITHREAD) {
    cout << "Running in Multithreading mode!" << endl;

        vector<int> result;
        for (int i = 0; i < threadsNumber; i++) {
            result.emplace_back(0);
        }

        for (double a = 0.05; a <= 1.0; a += 0.05) {
            for (double b = 0.05; b <= 1.0; b += 0.05) {
                for (double c = 0.05; c <= 1.0; c += 0.05) {
                    vector<thread> threads; 
                    for (int i = 0; i < threadsNumber; i++) {
                        threads.emplace_back(thread(calculatePointsOfFigure, a, b, c, &result[i]));
                    } 
                    for (auto& th : threads) {
                        th.join();
                    }    
                    testAllRandomPoints(a, b, c, result);
                
                }
            }
        }


    } else {
        cout << "Running in Non-Multithreading mode!" << endl;

        vector<int> result;
        for (int i = 0; i < threadsNumber; i++) {
            result.emplace_back(0);
        }

        for (double a = 0.05; a < 1; a += 0.05) {
            for (double b = 0.05; b < 1; b += 0.05) {
                for (double c = 0.05; c < 1; c += 0.05) {

                    for (int i = 0; i < threadsNumber; i++) {
                        calculatePointsOfFigure(a, b, c, &result[i]);

                    } 
                    testAllRandomPoints(a, b, c, result);
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
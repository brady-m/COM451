#include <iostream>
#include <thread>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <sched.h>
#include <chrono>

using namespace std;

typedef chrono::high_resolution_clock Time;
typedef chrono::duration<float> Duration;

int gNumThreads = 8;
double t = 0.001;
int iterations = 10000000;


double rand_coordinate(double min, double max){
    return min + ((double) rand() / RAND_MAX) * (max - min);
}

void calcPoints(int tid, double a, double b, double c, int &is_moving, cpu_set_t *mask) {

    if (mask != NULL) {
        sched_setaffinity(0, sizeof(mask), mask);
    }

    int initialCore = sched_getcpu() + 1;

    srand(static_cast<unsigned int>(clock()));
    
    double x = rand_coordinate(0.1, 10.0);
    double y = rand_coordinate(0.1, 10.0);
    double z = rand_coordinate(0.1, 10.0);
 
    tid++;
 
    for (int i = 0; i < iterations; i++ ) {
        double xt = t * (a * cos(y));
        double yt = t * (b + x - z);
        double zt = t * (c * (y - z));

        x += xt;
        y += yt;
        z += zt;

        if (xt*yt*zt == 0 || x+y+z > 10000){
            is_moving = 0;
            break;
        }
    }
}

bool check_moving(vector<int> &is_moving){
    for (int i = 0; i < is_moving.size(); i++)
        if (is_moving[i] == 0)
            return false;
    return true;
}

int calcAttractor(bool isMultithread) {
    int MULTITHREAD = (int) isMultithread; 

    unsigned numCPU = thread::hardware_concurrency();
    printf("Available CPU cores: %d\n", numCPU);

    if (MULTITHREAD)
        printf("multi-threading mode\n");
    else
        printf("no-multi-threading mode\n");

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset); 
    for (int i = 0; i < numCPU; i++) {
        CPU_SET(i, &cpuset);
    }

    auto start = Time::now();

    if (MULTITHREAD) {
        thread threads[gNumThreads];

        for (double varA = 0.05; varA <= 1.0; varA += 0.05)
        {
            for (double varB = 0.05; varB <= 1.0; varB += 0.05)
            {
                for (double varC = 0.05; varC <= 1.0; varC += 0.05)
                {
                    vector<int> is_moving(gNumThreads, 1);

                    for (int tid = 0; tid < gNumThreads; tid++) {
                        threads[tid] = thread(calcPoints, tid, varA, varB, varC, ref(is_moving[tid]), &cpuset);
                    }

                    for (int t_id = 0; t_id < gNumThreads; t_id++) threads[t_id].join();

                    if (check_moving(is_moving)){
                        printf("Good parameters: a = %.2f, b = %.2f, c = %.2f\n", varA, varB, varC);
                    }
                }   
            }
        }

    } else {
        for (double varA = 0.05; varA <= 1.0; varA += 0.05)
        {
            for (double varB = 0.05; varB <= 1.0; varB += 0.05)
            {
                for (double varC = 0.05; varC <= 1.0; varC += 0.05)
                {
                    vector<int> is_moving(gNumThreads, 1);

                    for (int tid = 0; tid < gNumThreads; tid++) {

                        calcPoints(tid, varA, varB, varC, ref(is_moving[tid]), NULL);
                    }
                    if (check_moving(is_moving)){
                        printf("Good parameters: a = %.2f, b = %.2f, c = %.2f\n", varA, varB, varC);
                    }
                }
            }
        }
    }

    auto finish = Time::now();
    Duration seconds = finish - start;

    printf("%f seconds\n", seconds.count());
}

/*int main(int argc, char *argv[]){
    int multithread = (argc == 2) && (bool) atoi(argv[1]);

    calcAttractor(multithread);
    return 0;
}*/
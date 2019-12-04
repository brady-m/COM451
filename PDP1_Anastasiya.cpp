#include <stdio.h>
#include <thread>

using namespace std;

double t = 0.0018;

double random_number() { return ((double)rand() / (RAND_MAX)); }

int runIt(int a, int b, int c, int* val) {
    *val = 1;

    int max = 1;
    int min = -1;
    float x = random_number();
    float y = random_number();
    float z = random_number();

    float dx, dy, dz;
    
    for (int i = 0; i < 10000000; i++) {
        dx = x + t * ((-a) * (x - y));
        dy = y + t * ((-x) * z + b * x - y);
        dz = z + t * (x * y - c * z);

        x += dx;
        y += dy;
        z += dz;
    }
    if (dx + dy + dz < 0.001) {
        *val = 0;
    }
    if (x + y + z > 50) {
        *val = 0;
    }
}

bool is_good(int vals[], int n) {
    bool isAll = true;
    for (int i = 0; i < n; i++) {
        if (vals[i] == 0) {
            isAll = false;
            break;
        }
    }
    return isAll;
}

int mainRun(int arg) {
    int number_of_threads = 8;
    int MULTITHREAD = 0;  // set default
    if (arg != 0) {
        MULTITHREAD = arg;
    }
    if (MULTITHREAD != 0) {
        number_of_threads = thread::hardware_concurrency();
        printf("Amount of threads is %d\n", number_of_threads);
    } else {
        number_of_threads = 8;
    }
    time_t theStart, theEnd;
    time(&theStart);

    int vals[number_of_threads];
    thread zThreads[number_of_threads];

    for (float a = 0.05; a < 1; a += 0.05) {
        for (float b = 0.05; b < 1; b += 0.05) {
            for (float c = 0.05; c < 1; c += 0.05) {
                for (int tid = 0; tid < number_of_threads; tid++) {
                    zThreads[tid] = thread(runIt, a, b, c, &vals[tid]);
                }
                for (int tid = 0; tid < number_of_threads; tid++) {
                    zThreads[tid].join();
                }
                if (is_good(vals, number_of_threads)) {
                    printf("GoodParams: %f, %f, %f\n", a, b, c);
                } else {
                    printf("Faild!\n");
                }
            }
        }
    }
    time(&theEnd);
    printf("MULTITHREADING seconds used: %ld\n", theEnd - theStart);
    free(zThreads);
    return 0;
}

/******************************************************************************/

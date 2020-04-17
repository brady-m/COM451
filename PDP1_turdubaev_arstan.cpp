#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>

#include "PDP1_turdubaev_arstan.h"

const double pi = std::acos(-1);

int MULTITHREADING_MODE = 1;
int NUMBER_OF_THREADS = 8;

int iteration_number = 10000000;
double t = 0.01;

double get_random(double min, double max) {
    /* Returns a random double between min and max */

    return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

// void search_attractor_params(int t_id, double a, double b, double c, std::vector<int>& is_moving)
void search_attractor_params(double a, double b, double c, int& is_moving)
{
    double x = get_random(0.1, 1.0);
    double y = get_random(0.1, 1.0);
    double z = get_random(0.1, 1.0);

    for (int i = 0; i < iteration_number; i++)
    {
        // Attractor #1
        double xt = t * (a * cos(y));
        double yt = t * (b + x - z);
        double zt = t * (c * (y - z));

        // // Embryo Attractor
        // double xt = t * ((z * y) / a);
        // double yt = t * (y - z + b);
        // double zt = t * (c + y);

        // // Attractor #3
        // double xt = t * ((z * y) / a);
        // double yt = t * (y - x + b);
        // double zt = t * (c + y);
    
        // // Lorrenz Attractor
        // double xt = t * a * (y - x);
        // double yt = t * (x * (b - z));
        // double zt = t * (x * y - c * z);

        // // Windmill Attractor
        // double xt = t * sin(cos(y));
        // double yt = t * sin(sin(cos(0.96)) / x);
        // double zt = t * sin(y - z);

        x += xt;
        y += yt;
        z += zt;
        
        if (xt * yt * zt == 0 || x + y + z > 10000)
        {
            // is_moving[t_id] = 0;
            is_moving = 0;
            // std::cout << "RIP: " << "x = " << x << ", y = " << y << ", z = " << z << std::endl;
            break;
        }


    }
}

bool check_params(std::vector<int>& is_moving)
{
    for (int i = 0; i < is_moving.size(); i++)
    {
        if (is_moving[i] == 0)
        {
            return false;
        }
    }

    return true;
}

void run_PDP1() {

    srand(static_cast<unsigned int>(clock()));

    auto start_time = std::chrono::high_resolution_clock::now();

    if (MULTITHREADING_MODE)
    {
        std::thread threads[NUMBER_OF_THREADS];

        for (double a = 0.05; a <= 1.0; a += 0.05)
        {
            for (double b = 0.05; b <= 1.0; b += 0.05)
            {
                for (double c = 0.05; c <= 1.0; c += 0.05)
                {
                    std::vector<int> is_moving(NUMBER_OF_THREADS, 1);

                    for (int t_id = 0; t_id < NUMBER_OF_THREADS; t_id++)
                    {
                        threads[t_id] = std::thread(search_attractor_params, a, b, c, std::ref(is_moving[t_id]));
                    }

                    for (int t_id = 0; t_id < NUMBER_OF_THREADS; t_id++)
                    {
                        threads[t_id].join();
                    }
                    if (check_params(is_moving))
                    {

                        std::cout << std::fixed;
                        std::cout << std::setprecision(2);
                        std::cout << "Good parameters: a = " << a << ", b = " << b << ", c = " << c << std::endl; 
                    }
                }   
            }
        }
    }

    else
    {
        for (double a = 0.05; a <= 1.0; a += 0.05)
        {
            for (double b = 0.05; b <= 1.0; b += 0.05)
            {
                for (double c = 0.05; c <= 1.0; c += 0.05)
                {
                    std::vector<int> is_moving(NUMBER_OF_THREADS, 1);

                    for (int i = 0; i < NUMBER_OF_THREADS; i++)
                    {
                        search_attractor_params(a, b, c, is_moving[i]);
                    }

                    if (check_params(is_moving))
                    {

                        std::cout << std::fixed;
                        std::cout << std::setprecision(2);
                        std::cout << "Good parameters: a = " << a << ", b = " << b << ", c = " << c << std::endl; 
                    }
                }
            }
        }
    }
    

    auto stop_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time); 

    std::cout << duration.count() / 1000.0 << " seconds" << std::endl;
}
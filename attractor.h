#ifndef calcAttractor
#define calcAttractor

#include<stdio.h>
#include <thread>
#include <math.h>
#include <vector>
using namespace std;


double fRand(double fMin, double fMax);
void testAllRandomPoints(double a, double b, double c, vector<int> result);
void calculatePointsOfFigure (double a, double b, double c, int* result);
int RunAssignment1(int);


#endif 
#ifndef hPDP1Lib
#define hPDP1Lib

#include <stdio.h>
#include <thread>
#include <vector>

using namespace std;

int runIt(int a, int b, int c, int* val);
bool check_moving(vector<int> &is_moving);
int calcAttractor(bool isMultithread);

#endif

#include <vector>

double get_random(double min, double max);

void search_attractor_params(double a, double b, double c, int& is_moving);

bool check_params(std::vector<int>& is_moving);

void run_PDP1();
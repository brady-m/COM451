#ifndef hSTRUCTPOINTLib
#define hSTRUCTPOINTLib

struct APoint{
	double       x,       y,       z;
  double delta_x, delta_y, delta_z;
  double start_x, start_y, start_z;
  int xIdx, yIdx;
  double red,           blue,           green;
  int color_heatTransfer;
};

#endif

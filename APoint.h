#ifndef hSTRUCTPOINTLib
#define hSTRUCTPOINTLib

enum Color {
  red = 0, green, blue
};

struct APoint{
  
	double x, y, z;
  double start_x, start_y, start_z;
  double changed_x, changed_y, changed_z;

  int xIdx, yIdx;

  double red, blue, green;

  Color color;

  void updateParametrs() {
    x += changed_x;
    y += changed_y;
    z += changed_z;
  }

  int get_color() {
        if ((this->red >= this->green) && (this->red >= this->blue))
        return Color::red;
    else if (this->green >= this->blue)
        return Color::green;
    else
        return Color::blue;
  }
};

#endif

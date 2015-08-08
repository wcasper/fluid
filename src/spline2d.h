#ifndef __SPLINE2D_H
#define __SPLINE2D_H

typedef struct {
  double p[4][4]; // spline data
} spline2d_cell_t;

typedef struct {
  int nx, ny;

  spline2d_cell_t *cells;

} spline2d_t;

spline2d_t spline2d_init(double *data, int m, int n);

void spline2d_free(spline2d_t *myspline);

double spline2d_eval(spline2d_t *myspline, int i, int j, double t1, double t2);

double spline2d_eval_xderiv(spline2d_t *myspline, int i, int j, double t1, double t2);

double spline2d_eval_yderiv(spline2d_t *myspline, int i, int j, double t1, double t2);

#endif


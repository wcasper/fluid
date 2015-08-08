#ifndef __TOPO_H
#define __TOPO_H

#include "spline2d.h"

// depth of water column at each grid point
extern double *topo_hb;
extern double *topo_d2b;
extern double *topo_refl_x;
extern double *topo_refl_y;
extern double *topo_refl_z;
extern int    *topo_refl_task;

// spline data for topography
extern spline2d_t topo_spline;

int topo_init();

int topo_finalize();

int topo_interpolate(double *qvert);

#endif


#ifndef __GRID_H
#define __GRID_H

#include <stddef.h>
#include <stdbool.h>

extern ptrdiff_t grid_nd;

extern ptrdiff_t grid_nx;
extern ptrdiff_t grid_ny;
extern ptrdiff_t grid_nz;

extern ptrdiff_t grid_nx_local;
extern ptrdiff_t grid_ny_local;
extern ptrdiff_t grid_nz_local;
extern ptrdiff_t grid_nn_local;
extern ptrdiff_t grid_n0_local;

extern double grid_lx;
extern double grid_ly;
extern double grid_lz;

extern double grid_dx;
extern double grid_dy;
extern double grid_dz;

extern double *grid_kx;
extern double *grid_ky;
extern double *grid_kz;

extern int *grid_ki;
extern int *grid_kj;
extern int *grid_kk;

extern bool *grid_dealias_mask;

extern double *grid_wgt;

int grid_init();
int grid_finalize();

#define GRID_TYPE_PHYSICAL 0
#define GRID_TYPE_SPECTRAL 1

#endif


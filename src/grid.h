#ifndef __GRID_H
#define __GRID_H

#include <stddef.h>
#include <stdbool.h>

extern ptrdiff_t grid_nd;

extern ptrdiff_t grid_nx;
extern ptrdiff_t grid_ny;
extern ptrdiff_t grid_nz;
extern ptrdiff_t grid_2d_nn;
extern ptrdiff_t grid_3d_nn;

extern ptrdiff_t grid_2d_nx_local;
extern ptrdiff_t grid_2d_nn_local;
extern ptrdiff_t grid_2d_n0_local;
extern ptrdiff_t grid_3d_nn_local;

extern double grid_lx;
extern double grid_ly;
extern double grid_lz;

extern double grid_dx;
extern double grid_dy;
extern double grid_dz;

extern double *grid_2d_kx;
extern double *grid_2d_ky;
extern double *grid_vd_kzo;
extern double *grid_vd_kze;

extern double grid_2d_ksq_max;
extern double grid_vd_ksq_max;

extern double *grid_2d_x;
extern double *grid_2d_y;
extern double *grid_vd_z;

extern int *grid_2d_i;
extern int *grid_2d_j;

extern int *grid_2d_ki;
extern int *grid_2d_kj;

extern bool *grid_2d_dealias_mask;
extern bool *grid_2d_buffer;

extern double *grid_2d_wgt;

int grid_init();
void grid_finalize();

typedef enum {
  GRID_VERTICAL_LAYOUT_PERIODIC,
  GRID_VERTICAL_LAYOUT_SINE,
  GRID_VERTICAL_LAYOUT_COSINE
} grid_vertical_layout_t;

extern grid_vertical_layout_t grid_vertical_layout;

#endif


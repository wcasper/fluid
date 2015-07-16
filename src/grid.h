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

typedef enum {
  GRID_TYPE_PHYSICAL, /// physical grid
  GRID_TYPE_SPECTRAL, /// spectral grid
} grid_type_t;

typedef enum {
  GRID_LAYOUT_2D_PP,  /// 2d doubly periodic
  GRID_LAYOUT_2D_PN,  /// 2d periodic/neumann
  GRID_LAYOUT_2D_NN,  /// 2d neumann/neumann
  GRID_LAYOUT_3D_PPP, /// 3d triply periodic
  GRID_LAYOUT_3D_PPN, /// 3d periodic/periodic/neumann
  GRID_LAYOUT_3D_PNN, /// 3d periodic/neumann/neumann
  GRID_LAYOUT_3D_NNN, /// 3d neumann/neumann/neumann
} grid_layout_t;

extern grid_layout_t grid_layout;

#endif


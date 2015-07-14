#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <stdlib.h>

#include "grid.h"

ptrdiff_t grid_nd = 2;

ptrdiff_t grid_nx = 256;
ptrdiff_t grid_ny = 256;
ptrdiff_t grid_nz = 256;

ptrdiff_t grid_nx_local = 0;
ptrdiff_t grid_ny_local = 0;
ptrdiff_t grid_nz_local = 0;
ptrdiff_t grid_nn_local = 0;
ptrdiff_t grid_n0_local = 0;

double grid_lx = 1.0;
double grid_ly = 1.0;
double grid_lz = 1.0;

double grid_dx = 0.0;
double grid_dy = 0.0;
double grid_dz = 0.0;

double *grid_kx 	= NULL;
double *grid_ky 	= NULL;
double *grid_kz 	= NULL;

int *grid_ki         = NULL;
int *grid_kj         = NULL;
int *grid_kk         = NULL;

double *grid_wgt	= NULL;

bool *grid_dealias_mask = NULL;

int grid_init_2d();
int grid_init_3d();

int grid_init() {
  if(grid_nd == 2) {
    grid_init_2d();
  }
  else if(grid_nd == 3) {
    grid_init_3d();
  }
  else {
    fprintf(stderr, "unsupported dimension\n");
  }

  return 0;
}

int grid_init_2d() {
  int i, j, idx, ki, kj;

  grid_nx = 256;
  grid_ny = 256;

  grid_lx = 1.0;
  grid_ly = 1.0;

  grid_dx = grid_lx/(double)grid_nx;
  grid_dy = grid_ly/(double)grid_ny;

  fftw_mpi_init();
  grid_nn_local
    = fftw_mpi_local_size_2d(grid_nx, grid_ny/2+1,
                             MPI_COMM_WORLD,
                             &grid_nx_local, &grid_n0_local);
  grid_ny_local = grid_ny;

  grid_kx = calloc(grid_nn_local, sizeof(double));
  grid_ky = calloc(grid_nn_local, sizeof(double));

  grid_ki = calloc(grid_nn_local, sizeof(int));
  grid_kj = calloc(grid_nn_local, sizeof(int));

  grid_dealias_mask = calloc(grid_nn_local, sizeof(bool));

  grid_wgt = calloc(grid_nn_local, sizeof(double));

  for (i = 0; i < grid_nx_local; i++) {
    for (j = 0; j < grid_ny_local/2 + 1; j++) {
      idx = i*(grid_ny_local/2 + 1) + j;

      ki = grid_n0_local + i;
      kj = j;
      if(2*ki > grid_nx) ki -= grid_nx;

      grid_ki[idx] = ki;
      grid_kj[idx] = kj;

      if(3*abs(ki) < grid_nx ||
         3*abs(kj) < grid_ny) {
        grid_dealias_mask[idx] = true;
      }

      if((2*ki)%grid_nx == 0 &&
         (2*kj)%grid_ny == 0) {
        grid_kx[idx] = 0.0;
        grid_ky[idx] = 0.0;
      }
      else{
        grid_kx[idx] = grid_dx*2.0*M_PI*ki;
        grid_ky[idx] = grid_dy*2.0*M_PI*kj;
      }

      if((2*j)%grid_ny == 0) {
        grid_wgt[idx] = 1.0;
      }
      else {
        grid_wgt[idx] = 2.0;
      }
    }
  }

  return 0;
}

int grid_init_3d() {
  int i, j, k, idx, ki, kj, kk;

  grid_nx = 256;
  grid_ny = 256;
  grid_nz = 256;

  grid_lx = 1.0;
  grid_ly = 1.0;
  grid_lz = 1.0;

  grid_dx = grid_lx/(double)grid_nx;
  grid_dy = grid_ly/(double)grid_ny;
  grid_dz = grid_lz/(double)grid_nz;

  fftw_mpi_init();
  grid_nn_local
    = fftw_mpi_local_size_3d(grid_nx, grid_ny, grid_nz/2+1,
                             MPI_COMM_WORLD,
                             &grid_nx_local, &grid_n0_local);
  grid_ny_local = grid_ny;
  grid_nz_local = grid_nz;

  grid_kx = calloc(grid_nn_local, sizeof(double));
  grid_ky = calloc(grid_nn_local, sizeof(double));
  grid_kz = calloc(grid_nn_local, sizeof(double));

  grid_ki = calloc(grid_nn_local, sizeof(int));
  grid_kj = calloc(grid_nn_local, sizeof(int));
  grid_kk = calloc(grid_nn_local, sizeof(int));

  grid_dealias_mask = calloc(grid_nn_local, sizeof(bool));

  grid_wgt = calloc(grid_nn_local, sizeof(double));

  for (i = 0; i < grid_nx_local; i++) {
    for (j = 0; j < grid_ny_local; j++) {
      for (k = 0; k < grid_nz_local/2 + 1; j++) {
        idx = i*(grid_ny_local)*(grid_nz_local/2 + 1)
            + j*(grid_nz_local/2 + 1) + k;

        ki = grid_n0_local + i;
        kj = j;
        kk = k;
        if(2*ki > grid_nx) ki -= grid_nx;
        if(2*kj > grid_ny) ki -= grid_ny;

        grid_ki[idx] = ki;
        grid_kj[idx] = kj;
        grid_kk[idx] = kk;

        if(3*abs(ki) < grid_nx ||
           3*abs(kj) < grid_ny ||
           3*abs(kk) < grid_nz) {
          grid_dealias_mask[idx] = true;
        }

        if((2*ki)%grid_nx == 0 &&
           (2*kj)%grid_ny == 0 &&
           (2*kk)%grid_nz == 0) {
          grid_kx[idx] = 0.0;
          grid_ky[idx] = 0.0;
          grid_kz[idx] = 0.0;
        }
        else{
          grid_kx[idx] = grid_dx*2.0*M_PI*ki;
          grid_ky[idx] = grid_dy*2.0*M_PI*kj;
          grid_kz[idx] = grid_dz*2.0*M_PI*kk;
        }

        if((2*k)%grid_nz == 0 && (i > 0 || j > 0)) {
          grid_wgt[idx] = 0.5;
        }
        else {
          grid_wgt[idx] = 1.0;
        }
      }
    }
  }

  return 0;
}

int grid_finalize() {
  free(grid_kx);
  free(grid_ky);
  if(grid_nd == 3) {
    free(grid_kz);
  }

  free(grid_ki);
  free(grid_kj);
  if(grid_nd == 3) {
    free(grid_kk);
  }

  free(grid_dealias_mask);
  free(grid_wgt);

  return 0;
}


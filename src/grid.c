#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <iniparser.h>

#include "grid.h"
#include "error.h"
#include "config.h"
#include "comm.h"

ptrdiff_t grid_nd = 2;

ptrdiff_t grid_nx = 64;
ptrdiff_t grid_ny = 64;
ptrdiff_t grid_nz = 64;
ptrdiff_t grid_2d_nn = 4096;
ptrdiff_t grid_3d_nn = 262144;

ptrdiff_t grid_2d_nx_local = 0;
ptrdiff_t grid_2d_nn_local = 0;
ptrdiff_t grid_2d_n0_local = 0;
ptrdiff_t grid_3d_nn_local = 0;

double grid_lx = 1.0;
double grid_ly = 1.0;
double grid_lz = 1.0;

double grid_dx = 0.0;
double grid_dy = 0.0;
double grid_dz = 0.0;

double *grid_2d_kx = NULL;
double *grid_2d_ky = NULL;
double *grid_vd_kzo = NULL;
double *grid_vd_kze = NULL;

double *grid_2d_x = NULL;
double *grid_2d_y = NULL;
double *grid_vd_z = NULL;

int *grid_2d_i = NULL;
int *grid_2d_j = NULL;

int *grid_2d_ki = NULL;
int *grid_2d_kj = NULL;

double grid_2d_ksq_max = 0.0;
double grid_vd_ksq_max = 0.0;

double *grid_2d_wgt	= NULL;

bool *grid_2d_dealias_mask = NULL;
bool *grid_2d_buffer = NULL;

static int grid_read_config();
static int grid_init_layout_2d();
static int grid_init_layout_3d();

int grid_read_config() {
  int status = 0;
  dictionary *dict;

  if(my_task == master_task) {
    // read the configuration file
    dict = iniparser_load(config_file_name);
    if(!dict) {
      status = 1;
    }
  }
  error_check(&status, "error reading config file\n");
  if(status) return status;
  
  if(my_task == master_task) {
    // read in grid initialization data
    grid_nd = iniparser_getint(dict, "grid:nd", grid_nd);
    switch(grid_nd) {
      case(2):
        grid_nx = iniparser_getint(dict, "grid:nx", grid_nx);
        grid_ny = iniparser_getint(dict, "grid:ny", grid_ny);
        grid_lx = iniparser_getdouble(dict, "grid:lx", grid_lx);
        grid_ly = iniparser_getdouble(dict, "grid:ly", grid_ly);
        break;
      case(3):
        grid_nx = iniparser_getint(dict, "grid:nx", grid_nx);
        grid_ny = iniparser_getint(dict, "grid:ny", grid_ny);
        grid_nz = iniparser_getint(dict, "grid:nz", grid_nz);
        grid_lx = iniparser_getdouble(dict, "grid:lx", grid_lx);
        grid_ly = iniparser_getdouble(dict, "grid:ly", grid_ly);
        grid_lz = iniparser_getdouble(dict, "grid:lz", grid_lz);
        break;
      default:
        status = 1;
        break;
    }
    iniparser_freedict(dict);
  }

  MPI_Bcast(&grid_nd,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_nx,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_ny,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_nz,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_nz,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_lx,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_ly,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_lz,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);

  error_check(&status, "bad nd value in config file\n");
  return status;
}

int grid_init() {
  int status = 0;

  status = grid_read_config();
  error_check(&status, "error in grid_read_config\n");
  if(status) return status;

  switch(grid_nd) {
    case 2:
      status = grid_init_layout_2d();
      error_check(&status, "error in grid_init_layout_2d\n");
      grid_3d_nn = grid_2d_nn;
      grid_3d_nn_local = grid_2d_nn_local;
      break;
    case 3:
      status = grid_init_layout_3d();
      error_check(&status, "error in grid_init_layout_3d\n");
      break;
    default:
      error_check(&status, "unsupported number of dimensions\n");
      status = 1;
      break;
  }

  return status;
}

int grid_init_layout_2d() {
  int i, j, idx, ki, kj;
  int status = 0;

  grid_dx = grid_lx/(double)grid_nx;
  grid_dy = grid_ly/(double)grid_ny;

  fftw_mpi_init();
  grid_2d_nn_local
    = fftw_mpi_local_size_2d(grid_nx, grid_ny/2+1,
                             MPI_COMM_WORLD,
                             &grid_2d_nx_local, &grid_2d_n0_local);

  grid_2d_nn = grid_nx*grid_ny;

  grid_2d_kx = calloc(grid_2d_nn_local, sizeof(double));
  grid_2d_ky = calloc(grid_2d_nn_local, sizeof(double));

  grid_2d_ki = calloc(grid_2d_nn_local, sizeof(int));
  grid_2d_kj = calloc(grid_2d_nn_local, sizeof(int));

  grid_2d_x  = calloc(grid_2d_nn_local*2, sizeof(double));
  grid_2d_y  = calloc(grid_2d_nn_local*2, sizeof(double));

  grid_2d_i  = calloc(grid_2d_nn_local*2, sizeof(int));
  grid_2d_j  = calloc(grid_2d_nn_local*2, sizeof(int));

  grid_2d_dealias_mask = calloc(grid_2d_nn_local, sizeof(bool));
  grid_2d_buffer = calloc(grid_2d_nn_local*2, sizeof(bool));

  grid_2d_wgt = calloc(grid_2d_nn_local, sizeof(double));

  for (i = 0; i < grid_2d_nx_local; i++) {
    for (j = 0; j < grid_ny/2 + 1; j++) {
      idx = i*(grid_ny/2 + 1) + j;

      ki = grid_2d_n0_local + i;
      kj = j;
      if(2*ki > grid_nx) ki -= grid_nx;
      ki *= -1;
      kj *= -1;

      grid_2d_ki[idx] = ki;
      grid_2d_kj[idx] = kj;

      if(3*abs(ki) < grid_nx &&
         3*abs(kj) < grid_ny) {
        grid_2d_dealias_mask[idx] = false;
      }
      else {
        grid_2d_dealias_mask[idx] = true;
      }

      if((2*abs(ki))%grid_nx == 0 &&
         (2*abs(kj))%grid_ny == 0) {
        grid_2d_kx[idx] = 0.0;
        grid_2d_ky[idx] = 0.0;
      }
      else if(2*j >= grid_ny) {
        grid_2d_kx[idx] = 0.0;
        grid_2d_ky[idx] = 0.0;
      }
      else{
        grid_2d_kx[idx] = 2.0*M_PI*ki/grid_lx;
        grid_2d_ky[idx] = 2.0*M_PI*kj/grid_ly;
      }

      // weight based on whether conjugate exists
      if((2*j)%grid_ny == 0) {
        grid_2d_wgt[idx] = 1.0;
      }
      else {
        grid_2d_wgt[idx] = 2.0;
      }
    }
  }

  grid_2d_ksq_max  = pow(2.0*M_PI*grid_nx*(1./3.)/grid_lx,2.0);
  grid_2d_ksq_max += pow(2.0*M_PI*grid_ny*(1./3.)/grid_ly,2.0);

  for(i = 0; i < grid_2d_nx_local; i++) {
    for(j = 0; j < (grid_ny/2 +1)*2; j++) {
      idx = i*(grid_ny/2 + 1)*2 + j;

        grid_2d_x[idx] = (grid_2d_n0_local + i)*grid_dx;
        grid_2d_y[idx] = j*grid_dy;

        grid_2d_i[idx] = grid_2d_n0_local + i;
        grid_2d_j[idx] = j;

      if(j >= grid_ny) {
        grid_2d_buffer[idx] = true;
      }
      else {
        grid_2d_buffer[idx] = false;
      }
    }
  }

  return status;
}

int grid_init_layout_3d() {
  int m, status;

  // initialize the 2d layout
  status = grid_init_layout_2d();
  error_check(&status, "error in grid_init_layout_2d\n");
  if(status) return status;

  // initialize the 3d layout
  grid_dz = grid_lz/(double)grid_nz;
  grid_3d_nn = grid_2d_nn*grid_nz;

  grid_vd_z  = calloc(grid_nz, sizeof(double));
  grid_vd_kzo = calloc(grid_nz+1, sizeof(double));
  grid_vd_kze = calloc(grid_nz+1, sizeof(double));

  grid_vd_ksq_max  = pow(1.0*M_PI*grid_nz*(2./3.)/grid_lz,2.0);

  for(m = 0; m < grid_nz; m++) {
    grid_vd_z[m]  = ((m+0.5)/(double)grid_nz)*grid_lz;
    grid_vd_kzo[m] =  M_PI*(m+1)/grid_lz;
    grid_vd_kze[m] = -M_PI*m/grid_lz;
  }
  grid_vd_kzo[grid_nz] = 0.0;
  grid_vd_kze[grid_nz] = 0.0;

  grid_3d_nn_local = grid_2d_nn_local*grid_nz;

  return 0;
}

void grid_finalize() {
  free(grid_2d_kx);
  free(grid_2d_ky);
  free(grid_2d_x);
  free(grid_2d_y);
  free(grid_2d_i);
  free(grid_2d_j);
  free(grid_2d_ki);
  free(grid_2d_kj);
  if(grid_nd == 3) {
    free(grid_vd_kzo);
    free(grid_vd_kze);
    free(grid_vd_z);
  }

  free(grid_2d_dealias_mask);
  free(grid_2d_buffer);
  free(grid_2d_wgt);
}


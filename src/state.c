#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <assert.h>

#include "state.h"
#include "grid.h"
#include "comm.h"

double *q;
double complex *kq;
ptrdiff_t nq = 1;

FILE *state_ifile,
     *state_ofile;

char ifile_name[256] = "init.ieeer8";

fftw_plan *p2s_plans, *s2p_plans;

double fft_normalization;

state_init_type_t state_init_type = STATE_INIT_TYPE_PATCHES_2D;

char * state_restart_file_name = NULL;

int state_init() {
  int n, idx;

  q  = calloc(grid_nn_local*2*nq,sizeof(double));
  kq = calloc(grid_nn_local*nq,sizeof(double complex));

  p2s_plans = calloc(nq,sizeof(fftw_plan));
  s2p_plans = calloc(nq,sizeof(fftw_plan));

  if(grid_nd == 2) {  
    fft_normalization = 1.0/(grid_nx*grid_ny);
  }
  else {
    fft_normalization = 1.0/(grid_nx*grid_ny*grid_nz);
  }

  /* create plan for out-of-place r2c DFT */
  for(n = 0; n < nq; n++) {
    idx = grid_nn_local*n;
    if(grid_nd == 2) {
      p2s_plans[n] =
        fftw_mpi_plan_dft_r2c_2d(grid_nx, grid_ny, &q[idx*2], &kq[idx],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
      s2p_plans[n] =
        fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, &kq[idx], &q[idx*2],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
    }
    else {
      p2s_plans[n] =
        fftw_mpi_plan_dft_r2c_3d(grid_nx, grid_ny, grid_nz,
                                 &q[idx*2], &kq[idx],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
      s2p_plans[n] =
        fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz,
                                 &kq[idx], &q[idx*2],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
    }
  }

  // Initialize spectral data
  switch(state_init_type) {
    case STATE_INIT_TYPE_RESTART:
      state_read(state_restart_file_name);
    break;
    case STATE_INIT_TYPE_PATCHES_2D:
      for(idx = 0; idx < grid_nn_local; idx++) {
        if(abs(grid_ki[idx]) == 1 &&
           abs(grid_kj[idx]) == 1 ){
          kq[idx] = (double)rand()/(double)RAND_MAX;
        }
      }
    break;
    case STATE_INIT_TYPE_PATCHES_3D:
      for(idx = 0; idx < grid_nn_local; idx++) {
        if(abs(grid_ki[idx]) == 1 &&
           abs(grid_kj[idx]) == 1 &&
           abs(grid_kk[idx]) == 1 ){
          kq[grid_nn_local*0 + idx] = (double)rand()/(double)RAND_MAX;
          kq[grid_nn_local*1 + idx] = (double)rand()/(double)RAND_MAX;
          kq[grid_nn_local*2 + idx] = grid_kx[idx]*kq[grid_nn_local*0 + idx]
                                    + grid_ky[idx]*kq[grid_nn_local*1 + idx];
          kq[grid_nn_local*2 + idx]/= grid_kz[idx]*(-1.0);
        }
      }
    break;
    default:
      fprintf(stderr, "unknown state initialization\n");
      return 1;
  }
  state_spectral2physical();

  return 0;
}

int state_physical2spectral() {
  ptrdiff_t n,idx;

  for(n = 0; n < nq; n++) {
    fftw_execute(p2s_plans[n]);
    for(idx = 0;  idx < grid_nn_local; idx++) {
      kq[idx] *= fft_normalization;
    }
  }

  return 0;
}

int state_spectral2physical() {
  ptrdiff_t n;

  for(n = 0; n < nq; n++) {
    fftw_execute(s2p_plans[n]);
  }

  return 0;
}

int state_read(char *ifile_name) {
  int n, idx;

  FILE *ifile;

  double complex *kq_global;

  size_t read_size = 0;

  if(grid_nd == 2) {
    read_size = grid_nx*(grid_ny/2 + 1);
  }
  else {
    read_size = grid_nx*grid_ny*(grid_nz/2 + 1);
  }

  if(my_task == master_task) {
    kq_global  = calloc(read_size, sizeof(double complex));

    ifile = fopen(ifile_name, "rb");
    assert(ifile);
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_nn_local;

    if(my_task == master_task) {
      fread(kq_global, sizeof(double complex), read_size, ifile);
    }

    scatter_global_array(&kq[idx], kq_global, sizeof(double complex),
                         GRID_TYPE_SPECTRAL);
  }

  if(my_task == master_task) {
    fclose(ifile);
    free(kq_global);
  }

  return 0;
}

int state_write(char *ofile_name) {
  int n, idx;

  FILE *ofile;

  double complex *kq_global;

  size_t write_size = 0;

  if(grid_nd == 2) {
    write_size = grid_nx*(grid_ny/2 + 1);
  }
  else {
    write_size = grid_nx*grid_ny*(grid_nz/2 + 1);
  }

  if(my_task == master_task) {
    kq_global = calloc(write_size, sizeof(double complex));
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name, "wb");
    assert(ofile);
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_nn_local;

    gather_global_array(&kq[idx], kq_global,
                        sizeof(double complex), GRID_TYPE_SPECTRAL);

    if(my_task == master_task) {
      fwrite(kq_global, sizeof(double complex), write_size, ofile);
    }
  }

  if(my_task == master_task) {
    fclose(ofile);
    free(kq_global);
  }

  return 0;
}

int state_finalize() {
  ptrdiff_t k;

  free(q);
  free(kq);

  for(k = 0; k < nq; k++) {
    fftw_destroy_plan(p2s_plans[k]);
    fftw_destroy_plan(s2p_plans[k]);
  }
  free(p2s_plans);
  free(s2p_plans);

  if(state_restart_file_name) free(state_restart_file_name);

  return 0;
}


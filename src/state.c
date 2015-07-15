#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>

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

int state_init(int numq) {
  int k, idx;

  nq = numq;
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
  for(k = 0; k < nq; k++) {
    idx = grid_nn_local*k;
    if(grid_nd == 2) {
      p2s_plans[k] =
        fftw_mpi_plan_dft_r2c_2d(grid_nx, grid_ny, &q[idx*2], &kq[idx],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
      s2p_plans[k] =
        fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, &kq[idx], &q[idx*2],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
    }
    else {
      p2s_plans[k] =
        fftw_mpi_plan_dft_r2c_3d(grid_nx, grid_ny, grid_nz,
                                 &q[idx*2], &kq[idx],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
      s2p_plans[k] =
        fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz,
                                 &kq[idx], &q[idx*2],
                                 MPI_COMM_WORLD, FFTW_MEASURE);
    }
  }

  // Initialize spectral data
//  state_read("infile.bin");
  if(my_task == master_task) {
    kq[1] = 1.0;
    kq[(grid_ny_local/2 + 1) + 1] = 1.0;
  }
  state_spectral2physical();

  return 0;
}

int state_physical2spectral() {
  ptrdiff_t k,idx;

  for(k = 0; k < nq; k++) {
    fftw_execute(p2s_plans[k]);
    for(idx = 0;  idx < grid_nn_local; idx++) {
      kq[idx] *= fft_normalization;
    }
  }

  return 0;
}

int state_spectral2physical() {
  ptrdiff_t k;

  for(k = 0; k < nq; k++) {
    fftw_execute(s2p_plans[k]);
  }

  return 0;
}

int state_read(char *ifile_name) {
  int n, idx;

  FILE *ifile;

  complex double *kq_global;

  size_t read_size = 0;

  if(grid_nd == 2) {
    read_size = grid_nx*(grid_ny/2 + 1);
  }
  else {
    read_size = grid_nx*grid_ny*(grid_nz/2 + 1);
  }

  if(my_task == master_task) {
    kq_global  = calloc(read_size, sizeof(complex double));

    ifile = fopen(ifile_name, "rb");
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_nn_local;

    if(my_task == master_task) {
      fread(kq_global, sizeof(complex double), read_size, ifile);
    }

    scatter_global_array(&kq[idx], kq_global, sizeof(complex double),
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

  complex double *kq_global;

  size_t write_size = 0;

  if(grid_nd == 2) {
    write_size = grid_nx*(grid_ny/2 + 1);
  }
  else {
    write_size = grid_nx*grid_ny*(grid_nz/2 + 1);
  }

  if(my_task == master_task) {
    kq_global = calloc(write_size, sizeof(complex double));
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name, "wb");
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_nn_local;

    gather_global_array(&kq[idx], kq_global,
                        sizeof(complex double), GRID_TYPE_SPECTRAL);

    if(my_task == master_task) {
      fwrite(kq, sizeof(complex double), write_size, ofile);
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

  return 0;
}


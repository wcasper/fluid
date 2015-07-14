#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>

#include "ins2d.h"
#include "grid.h"
#include "comm.h"
#include "state.h"
#include "diag.h"

double ins2d_kvisc    = 0.0; // kinematic viscosity

int ins2d_jac(double complex *kjac, double complex *kq_in);

double ins2d_step_rk4_adaptive(double dt, double err_bnd);

double *rwork1, *rwork2, *rwork3, *rwork4;

double complex *cwork1, *cwork2, *cwork3, *cwork4; 

fftw_plan  iplan1, iplan2, iplan3, iplan4, plan1;

int ins2d_init() {
  rwork1 = calloc(grid_nn_local*2,sizeof(double));
  rwork2 = calloc(grid_nn_local*2,sizeof(double));
  rwork3 = calloc(grid_nn_local*2,sizeof(double));
  rwork4 = calloc(grid_nn_local*2,sizeof(double));

  cwork1 = calloc(grid_nn_local,sizeof(double complex));
  cwork2 = calloc(grid_nn_local,sizeof(double complex));
  cwork3 = calloc(grid_nn_local,sizeof(double complex));
  cwork4 = calloc(grid_nn_local,sizeof(double complex));

  plan1 =
    fftw_mpi_plan_dft_r2c_2d(grid_nx, grid_ny, rwork1, cwork1,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan1 =
    fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, cwork1, rwork1,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan2 =
    fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, cwork2, rwork2,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan3 =
    fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, cwork3, rwork3,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan4 =
    fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, cwork4, rwork4,
                             MPI_COMM_WORLD, FFTW_MEASURE);

  return 0;
}

int ins2d_finalize() {
  fftw_destroy_plan(iplan1);
  fftw_destroy_plan(iplan2);
  fftw_destroy_plan(iplan3);
  fftw_destroy_plan(iplan4);
  fftw_destroy_plan(plan1);

  free(rwork1);
  free(rwork2);
  free(rwork3);
  free(rwork4);

  free(cwork1);
  free(cwork2);
  free(cwork3);
  free(cwork4);

  return 0;
}

// nonlinear part -- careful, careful!
int ins2d_jac(double complex *kjac, double complex *kq_in) {
  ptrdiff_t idx;

  double kx, ky;

  double normalization = 1.0/(grid_nx*grid_ny);

  for(idx = 0; idx < grid_nn_local; idx++) {
    if(grid_dealias_mask[idx]) {
      if(fabs(grid_kx[idx]) > 1e-14 || fabs(grid_ky[idx]) > 1e-14) {
        kx = grid_kx[idx];
        ky = grid_ky[idx];

        cwork1[idx] = -I*(kx*kq_in[idx])/(kx*kx + ky*ky);
        cwork2[idx] =  I*ky*kq_in[idx];
        cwork3[idx] =  I*kx*kq_in[idx];
        cwork4[idx] = -I*(ky*kq_in[idx])/(kx*kx + ky*ky);
      }
      else{
        cwork1[idx] = 0.0 + 0.0*I;
        cwork2[idx] = 0.0 + 0.0*I;
        cwork3[idx] = 0.0 + 0.0*I;
        cwork4[idx] = 0.0 + 0.0*I;
      }
    }
    else {
      cwork1[idx] = 0.0 + 0.0*I;
      cwork2[idx] = 0.0 + 0.0*I;
      cwork3[idx] = 0.0 + 0.0*I;
      cwork4[idx] = 0.0 + 0.0*I;
    }
  }

  fftw_execute(iplan1);
  fftw_execute(iplan2);
  fftw_execute(iplan3);
  fftw_execute(iplan4);

  for(idx = 0; idx < grid_nn_local*2; idx++) {
    rwork1[idx]  = rwork1[idx]*rwork2[idx];
    rwork1[idx] -= rwork3[idx]*rwork4[idx];
  }

  fftw_execute(plan1);

  for(idx = 0; idx < grid_nn_local; idx++) {
    kjac[idx] = cwork1[idx]*normalization;
  }

  return 0;
}

// Cash-Karp method of adaptive rk4
double ins2d_step_rk4_adaptive(double dt, double err_bnd) {
  ptrdiff_t ai, bi, idx;

  double kx, ky;

  double complex *ks, *kjac;

  double err_max = 0.0,
         err_avg = 0.0,
         err, err_max_global;

  int    err_cnt = 0;

  const int runge_kutta_num = 6;

  const double a[5][5] =
  {{1./5.       , 0.0      , 0.0        , 0.0           , 0.0       },
   {3./40.      , 9./40.   , 0.0        , 0.0           , 0.0       },
   {3./10.      , -9./10.  , 6./5.      , 0.0           , 0.0       },
   {-11./54.    , 5./2.    , -70./27.   , 35./27.       , 0.0       },
   {1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.}};
 
  //const double c[5] = {1./5.,3./10., 3./5., 1., 7./8.};

  const double b[6] = {37./378.,  0., 250./621.,
                       125./594., 0., 512./1771.};

  const double d[6] = {2825./27648., 0.,
                       18575./48384.,
                       13525./55296.,
                       277/14336., 0.25};

  ks   = calloc(grid_nn_local*runge_kutta_num,sizeof(double complex));
  kjac = calloc(grid_nn_local,   sizeof(double complex));

  for(bi = 0; bi < runge_kutta_num; bi++) {
    for(idx = 0; idx < grid_nn_local; idx++) {
        ks[grid_nn_local*bi + idx] = kq[idx];

        for(ai = 1; ai <= bi; ai++) {
          ks[grid_nn_local*bi + idx] +=
            dt*a[bi-1][ai-1]*ks[grid_nn_local*(ai-1) + idx];
        }
    }
    ins2d_jac(kjac, &ks[grid_nn_local*bi]);

    for(idx = 0; idx < grid_nn_local; idx++) {
      kx = grid_kx[idx];
      ky = grid_ky[idx];

      ks[grid_nn_local*bi + idx] *= -(kx*kx + ky*ky);
      ks[grid_nn_local*bi + idx] *= ins2d_kvisc;
      ks[grid_nn_local*bi + idx] -= kjac[idx];
    }
  }

  for(idx = 0; idx < grid_nn_local; idx++) {
    if(grid_dealias_mask[idx]) {
      for(bi = 0; bi < runge_kutta_num; bi++) {
        err = dt*(b[bi]-d[bi])*ks[grid_nn_local*bi + idx];
        if (fabs(err) > err_max) err_max = err;
        err_avg += fabs(err);
        err_cnt++;
      }
    }
  }
  err_avg /= err_cnt;

  MPI_Reduce(&err_max,  &err_max_global,  1,
             MPI_DOUBLE, MPI_MAX, master_task, MPI_COMM_WORLD);

  MPI_Bcast(&err_max_global, 1, MPI_DOUBLE,
            master_task, MPI_COMM_WORLD);

  if(err_max_global < err_bnd) {
    for(idx = 0; idx < grid_nn_local; idx++) {
      for(bi = 0; bi < runge_kutta_num; bi++) {
        kq[idx] += dt*b[bi]*ks[grid_nn_local*bi + idx];
      }
    }

    // dealias kq
    for(idx = 0; idx < grid_nn_local; idx++) {
      if(!grid_dealias_mask[idx]) {
        kq[idx] = 0.0;
      }
    }
  }

  free(ks);
  free(kjac);
  return err_max_global;
}


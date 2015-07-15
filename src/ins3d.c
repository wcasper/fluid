#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>

#include "ins3d.h"
#include "grid.h"
#include "comm.h"
#include "state.h"
#include "diag.h"

double ins3d_kvisc    = 0.0; // kinematic viscosity

int ins3d_adv(double complex *kadv, double complex *kq_in);

double ins3d_step_rk4_adaptive(double dt, double err_bnd_global);

double *rwork1, *rwork2, *rwork3,
       *rwork4, *rwork5, *rwork6;

double complex *cwork1, *cwork2, *cwork3,
               *cwork4, *cwork5, *cwork6; 

fftw_plan  iplan1, iplan2, iplan3,
           iplan4, iplan5, iplan6, plan1;

int ins3d_init() {
  rwork1 = calloc(grid_nn_local*2,sizeof(double));
  rwork2 = calloc(grid_nn_local*2,sizeof(double));
  rwork3 = calloc(grid_nn_local*2,sizeof(double));
  rwork4 = calloc(grid_nn_local*2,sizeof(double));
  rwork5 = calloc(grid_nn_local*2,sizeof(double));
  rwork6 = calloc(grid_nn_local*2,sizeof(double));

  cwork1 = calloc(grid_nn_local,sizeof(double complex));
  cwork2 = calloc(grid_nn_local,sizeof(double complex));
  cwork3 = calloc(grid_nn_local,sizeof(double complex));
  cwork4 = calloc(grid_nn_local,sizeof(double complex));
  cwork5 = calloc(grid_nn_local,sizeof(double complex));
  cwork6 = calloc(grid_nn_local,sizeof(double complex));

  plan1 =
    fftw_mpi_plan_dft_r2c_3d(grid_nx, grid_ny, grid_nz, rwork1, cwork1,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan1 =
    fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz, cwork1, rwork1,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan2 =
    fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz, cwork2, rwork2,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan3 =
    fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz, cwork3, rwork3,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan4 =
    fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz, cwork4, rwork4,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan5 =
    fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz, cwork5, rwork5,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  iplan6 =
    fftw_mpi_plan_dft_c2r_3d(grid_nx, grid_ny, grid_nz, cwork6, rwork6,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  return 0;
}

int ins3d_finalize() {
  fftw_destroy_plan(iplan1);
  fftw_destroy_plan(iplan2);
  fftw_destroy_plan(iplan3);
  fftw_destroy_plan(iplan4);
  fftw_destroy_plan(iplan5);
  fftw_destroy_plan(iplan6);
  fftw_destroy_plan(plan1);

  free(rwork1);
  free(rwork2);
  free(rwork3);
  free(rwork4);
  free(rwork5);
  free(rwork6);

  free(cwork1);
  free(cwork2);
  free(cwork3);
  free(cwork4);
  free(cwork5);
  free(cwork6);

  return 0;
}

// nonlinear part -- careful, careful!
int ins3d_adv(double complex *kadv, double complex *kq_in) {
  ptrdiff_t idx, n;

  double kx, ky, kz;

  double normalization = 1.0/(grid_nx*grid_ny*grid_nz);

  for(n = 0; n < 3; n++) {
    for(idx = 0; idx < grid_nn_local; idx++) {
      if(grid_dealias_mask[idx]) {
        if(fabs(grid_kx[idx]) > 1e-14 ||
           fabs(grid_ky[idx]) > 1e-14 ||
           fabs(grid_kz[idx]) > 1e-14) {
          kx = grid_kx[idx];
          ky = grid_ky[idx];
          kz = grid_kz[idx];

          cwork1[idx] =  I*kx*kq_in[idx + grid_nn_local*n];
          cwork2[idx] =  I*ky*kq_in[idx + grid_nn_local*n];
          cwork3[idx] =  I*kz*kq_in[idx + grid_nn_local*n];
          cwork4[idx] =  kq_in[idx + grid_nn_local*0];
          cwork5[idx] =  kq_in[idx + grid_nn_local*1];
          cwork6[idx] =  kq_in[idx + grid_nn_local*2];
        }
        else{
          cwork1[idx] = 0.0 + 0.0*I;
          cwork2[idx] = 0.0 + 0.0*I;
          cwork3[idx] = 0.0 + 0.0*I;
          cwork4[idx] = 0.0 + 0.0*I;
          cwork5[idx] = 0.0 + 0.0*I;
          cwork6[idx] = 0.0 + 0.0*I;
        }
      }
      else {
        cwork1[idx] = 0.0 + 0.0*I;
        cwork2[idx] = 0.0 + 0.0*I;
        cwork3[idx] = 0.0 + 0.0*I;
        cwork4[idx] = 0.0 + 0.0*I;
        cwork5[idx] = 0.0 + 0.0*I;
        cwork6[idx] = 0.0 + 0.0*I;
      }
    }

    fftw_execute(iplan1);
    fftw_execute(iplan2);
    fftw_execute(iplan3);
    fftw_execute(iplan4);
    fftw_execute(iplan5);
    fftw_execute(iplan6);

    for(idx = 0; idx < grid_nn_local*2; idx++) {
      rwork1[idx] = rwork1[idx]*rwork4[idx]
                  + rwork2[idx]*rwork5[idx]
                  + rwork3[idx]*rwork6[idx];
    }

    fftw_execute(plan1);

    for(idx = 0; idx < grid_nn_local; idx++) {
      kadv[idx + n*grid_nn_local] = cwork1[idx]*normalization;
    }
  }

  return 0;
}

// Cash-Karp method of adaptive rk4
double ins3d_step_rk4_adaptive(double dt, double err_bnd_global) {
  ptrdiff_t ai, bi, n, idx;

  double kx, ky, kz;

  double complex *ks, *kadv, kp;

  double err_max = 0.0,
         err, err_max_global;

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

  ks   = calloc(grid_nn_local*nq*runge_kutta_num,sizeof(double complex));
  kadv = calloc(grid_nn_local*nq,                sizeof(double complex));

  // calculate the runge kutta coefficients (ks)
  for(bi = 0; bi < runge_kutta_num; bi++) {
    for(idx = 0; idx < grid_nn_local; idx++) {
      for(n = 0; n < nq; n++) {
        ks[grid_nn_local*(nq*bi + n) + idx] = kq[grid_nn_local*n + idx];

        for(ai = 1; ai <= bi; ai++) {
          ks[grid_nn_local*(nq*bi + n) + idx] +=
            dt*a[bi-1][ai-1]*ks[grid_nn_local*(nq*(ai-1) + n) + idx];
        }
      }
    }
    ins3d_adv(kadv, &ks[grid_nn_local*nq*bi]);

    // calculate the intermediate velocity field ignoring pressure
    for(idx = 0; idx < grid_nn_local; idx++) {
      kx = grid_kx[idx];
      ky = grid_ky[idx];
      kz = grid_kz[idx];

      for(n = 0; n < 3; n++) {
        ks[grid_nn_local*(bi*nq + n) + idx] *= -(kx*kx + ky*ky + kz*kz);
        ks[grid_nn_local*(bi*nq + n) + idx] *= ins3d_kvisc;
        ks[grid_nn_local*(bi*nq + n) + idx] -= kadv[grid_nn_local*n + idx];
      }
    }

    // adjust velocity based on pressure forcing
    for(idx = 0; idx < grid_nn_local; idx++) {
      kx = grid_kx[idx];
      ky = grid_ky[idx];
      kz = grid_kz[idx];

      if(fabs(kx) > 1e-14 ||
         fabs(ky) > 1e-14 ||
         fabs(kz) > 1e-14 ){
        kp = ks[grid_nn_local*(bi*nq + 0) + idx]*kx*I
           + ks[grid_nn_local*(bi*nq + 1) + idx]*ky*I
           + ks[grid_nn_local*(bi*nq + 2) + idx]*kz*I;
        kp/= -(kx*kx + ky*ky + kz*kz);
      }

      ks[grid_nn_local*(bi*nq + 0) + idx] -= I*kx*kp;
      ks[grid_nn_local*(bi*nq + 1) + idx] -= I*ky*kp;
      ks[grid_nn_local*(bi*nq + 2) + idx] -= I*kz*kp;
    }
  }

  // calculate the maximum error
  for(idx = 0; idx < grid_nn_local; idx++) {
    if(grid_dealias_mask[idx]) {
      for(n = 0; n < nq; n++) {
        err = 0.0;
        for(bi = 0; bi < runge_kutta_num; bi++) {
          err += dt*(b[bi]-d[bi])*ks[grid_nn_local*(bi*nq + n) + idx];
        }
        err = fabs(err);
        if (err > err_max) err_max = err;
      }
    }
  }

  MPI_Reduce(&err_max,  &err_max_global,  1,
             MPI_DOUBLE, MPI_MAX, master_task, MPI_COMM_WORLD);

  MPI_Bcast(&err_max_global, 1, MPI_DOUBLE,
            master_task, MPI_COMM_WORLD);

  if(err_max_global < err_bnd_global) {
    for(idx = 0; idx < grid_nn_local; idx++) {
      for(bi = 0; bi < runge_kutta_num; bi++) {
        for(n = 0; n < nq; n++) {
          kq[grid_nn_local*n + idx] += dt*b[bi]*ks[grid_nn_local*(bi*nq + n) + idx];
        }
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
  free(kadv);
  return err_max_global;
}


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iniparser.h>

#include "ins2d.h"
#include "grid.h"
#include "comm.h"
#include "error.h"
#include "fourier.h"
#include "state.h"
#include "config.h"

double ins2d_kvisc    = 0.0;	// kinematic viscosity

double *rwork1, *rwork2, *rwork3, *rwork4;

double complex *cwork1, *cwork2, *cwork3, *cwork4;

static void ins2d_adv(double complex *kadv, double complex *kq_in);
static void ins2d_rhs(double complex *krhs, double complex *kstate);
static int ins2d_read_config();

int ins2d_read_config() {
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
    // read in ins2d initialization data
    ins2d_kvisc = iniparser_getdouble(dict, "ins2d:kvisc", ins2d_kvisc);
    iniparser_freedict(dict);
  }

  MPI_Bcast(&ins2d_kvisc,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);

  return status;
}

int ins2d_init() {
  int idx;

  int status = 0;

  // sanity check
  status = (nq != 1);
  error_check(&status, "wrong number of physical variables\n");
  if(status) return status;

  status = (grid_nd != 2);
  error_check(&status, "wrong number of dimensions\n");
  if(status) return status;

  // read in config file
  status = ins2d_read_config();
  error_check(&status, "error in ins2d_read_config\n");
  if(status) return status;

  // initialize work arrays
  rwork1 = fftw_alloc_real(grid_2d_nn_local*2);
  rwork2 = fftw_alloc_real(grid_2d_nn_local*2);
  rwork3 = fftw_alloc_real(grid_2d_nn_local*2);
  rwork4 = fftw_alloc_real(grid_2d_nn_local*2);

  cwork1 = fftw_alloc_complex(grid_2d_nn_local);
  cwork2 = fftw_alloc_complex(grid_2d_nn_local);
  cwork3 = fftw_alloc_complex(grid_2d_nn_local);
  cwork4 = fftw_alloc_complex(grid_2d_nn_local);

  for(idx = 0; idx < grid_2d_nn_local*2; idx++) {
    rwork1[idx] = 0.0;
    rwork2[idx] = 0.0;
    rwork3[idx] = 0.0;
    rwork4[idx] = 0.0;
  }

  for(idx = 0; idx < grid_2d_nn_local; idx++) {
    cwork1[idx] = 0.0 + 0.0*I;
    cwork2[idx] = 0.0 + 0.0*I;
    cwork3[idx] = 0.0 + 0.0*I;
    cwork4[idx] = 0.0 + 0.0*I;
  }

  return 0;
}

int ins2d_finalize() {
  fftw_free(rwork1);
  fftw_free(rwork2);
  fftw_free(rwork3);
  fftw_free(rwork4);

  fftw_free(cwork1);
  fftw_free(cwork2);
  fftw_free(cwork3);
  fftw_free(cwork4);

  return 0;
}

// nonlinear part -- careful, careful!
void ins2d_adv(double complex *kadv, double complex *kq_in) {
  ptrdiff_t idx2d;

  double kx, ky;

  // calculate the advection
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    kx = grid_2d_kx[idx2d];
    ky = grid_2d_ky[idx2d];

    // dealias
    if(grid_2d_dealias_mask[idx2d]) {
      cwork1[idx2d] = 0.0 + 0.0*I;
      cwork2[idx2d] = 0.0 + 0.0*I;
      cwork3[idx2d] = 0.0 + 0.0*I;
      cwork4[idx2d] = 0.0 + 0.0*I;
    }
    else {
      if(fabs(kx) > 1e-14 || fabs(ky) > 1e-14) {
        cwork1[idx2d] = -I*(kx*kq_in[idx2d])/(kx*kx + ky*ky);
        cwork2[idx2d] =  I*ky*kq_in[idx2d];
        cwork3[idx2d] =  I*kx*kq_in[idx2d];
        cwork4[idx2d] = -I*(ky*kq_in[idx2d])/(kx*kx + ky*ky);
      }
      else {
        cwork1[idx2d] = 0.0 + 0.0*I;
        cwork2[idx2d] = 0.0 + 0.0*I;
        cwork3[idx2d] = 0.0 + 0.0*I;
        cwork4[idx2d] = 0.0 + 0.0*I;
      }
    }
  }

  spectral2physical(cwork1, rwork1, 0);
  spectral2physical(cwork2, rwork2, 0);
  spectral2physical(cwork3, rwork3, 0);
  spectral2physical(cwork4, rwork4, 0);

  for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
    rwork1[idx2d] = rwork1[idx2d]*rwork2[idx2d]
                  - rwork3[idx2d]*rwork4[idx2d];
  }

  physical2spectral(rwork1, kadv, 0);
}

void ins2d_rhs(double complex *krhs, double complex *kstate) {
  ptrdiff_t idx2d;

  // calculate the advection
  ins2d_adv(krhs, kstate);

  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    krhs[idx2d] *= -1.0;
  }
}

// Cash-Karp method of adaptive rk4
double ins2d_step_rk4_adaptive(double dt, double err_bnd_global) {
  ptrdiff_t ai, bi, idx2d;

  double complex *ks, *krhs;

  double ksq;

  double err_max = 0.0,
         err, err_max_global;

  double complex kerr;

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

  ks   = fftw_alloc_complex(grid_2d_nn_local*runge_kutta_num);
  krhs = fftw_alloc_complex(grid_2d_nn_local);

  // calculate the runge kutta coefficients (ks)
  for(bi = 0; bi < runge_kutta_num; bi++) {
    for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
      ks[grid_2d_nn_local*bi + idx2d] = kq[idx2d];

      for(ai = 1; ai <= bi; ai++) {
        ks[grid_2d_nn_local*bi + idx2d] +=
          dt*a[bi-1][ai-1]*ks[grid_2d_nn_local*(ai-1) + idx2d];
      }
    }

    // calculate rhs
    ins2d_rhs(krhs, &ks[grid_2d_nn_local*bi]);
    for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
      ks[grid_2d_nn_local*bi + idx2d] = krhs[idx2d];
    }

  }  // end of loop over bi

  // calculate the maximum error
  err_max = 0.0;
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    if(!grid_2d_dealias_mask[idx2d]) {
      kerr = 0.0;
      for(bi = 0; bi < runge_kutta_num; bi++) {
        kerr += dt*(b[bi]-d[bi])*ks[grid_2d_nn_local*bi + idx2d];
      }
      err = cabs(kerr);
      if (err > err_max) err_max = err;
    }
  }

  MPI_Reduce(&err_max,  &err_max_global,  1,
             MPI_DOUBLE, MPI_MAX, master_task, MPI_COMM_WORLD);

  MPI_Bcast(&err_max_global, 1, MPI_DOUBLE,
            master_task, MPI_COMM_WORLD);

  if (err_max_global > err_bnd_global) {
    return err_max_global;
  }

  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(bi = 0; bi < runge_kutta_num; bi++) {
      kq[idx2d]
        += dt*b[bi]*ks[grid_2d_nn_local*bi + idx2d];
    }
  }

  // apply viscous damping (leapfrog of viscous forcing)
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    ksq = grid_2d_kx[idx2d]*grid_2d_kx[idx2d]
        + grid_2d_ky[idx2d]*grid_2d_ky[idx2d];
    kq[idx2d] *= exp(-ins2d_kvisc*dt*pow(ksq,4));
  }

  // dealias kq
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    if(grid_2d_dealias_mask[idx2d]) {
      kq[idx2d] = 0.0;
    }
  }

  fftw_free(ks);
  fftw_free(krhs);
  return err_max_global;
}




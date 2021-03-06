#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iniparser.h>

#include "ins3d.h"
#include "fluid.h"
#include "grid.h"
#include "comm.h"
#include "error.h"
#include "topo.h"
#include "fourier.h"
#include "state.h"
#include "config.h"

fluid_real ins3d_kvisc    = 0.0;	// kinematic viscosity
fluid_real ins3d_fcor     = 0.0;	// coriolis parameter
fluid_real ins3d_bfreq    = 0.0;	// coriolis parameter
fluid_real ins3d_sigma;		// boundary relax scale

fluid_real *rwork1, *rwork2, *rwork3,
       *rwork4, *rwork5, *rwork6, *rwork7;

fluid_complex *cwork1, *cwork2, *cwork3,
               *cwork4, *cwork5, *cwork6, *cwork7; 

static void ins3d_adv(fluid_complex *kadv, fluid_complex *kstate);
static void ins3d_rhs(fluid_complex *krhs, fluid_complex *kstate);
static void ins3d_p_adjust(fluid_complex *kstate);

static int ins3d_read_config();

int ins3d_read_config() {
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
    // read in ins3d initialization data
    ins3d_fcor  = iniparser_getdouble(dict, "ins3d:fcor",  ins3d_fcor);
    ins3d_kvisc = iniparser_getdouble(dict, "ins3d:kvisc", ins3d_kvisc);
    ins3d_bfreq = iniparser_getdouble(dict, "ins3d:bfreq", ins3d_kvisc);
    iniparser_freedict(dict);
  }

  MPI_Bcast(&ins3d_fcor,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&ins3d_kvisc,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&ins3d_bfreq,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);

  return status;
}

int ins3d_init() {
  int idx;

  int status = 0;

  // sanity check
  status = (nq != 4);
  error_check(&status, "wrong number of physical variables\n");
  if(status) return status;

  // sanity check
  status = (grid_nd != 3);
  error_check(&status, "wrong number of dimensions\n");
  if(status) return status;

  // read in config file
  status = ins3d_read_config();
  error_check(&status, "error in ins3d_read_config\n");
  if(status) return status;

  // initialize work arrays
  rwork1 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork2 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork3 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork4 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork5 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork6 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork7 = fftw_alloc_real(grid_3d_nn_local*2);

  cwork1 = fftw_alloc_complex(grid_3d_nn_local);
  cwork2 = fftw_alloc_complex(grid_3d_nn_local);
  cwork3 = fftw_alloc_complex(grid_3d_nn_local);
  cwork4 = fftw_alloc_complex(grid_3d_nn_local);
  cwork5 = fftw_alloc_complex(grid_3d_nn_local);
  cwork6 = fftw_alloc_complex(grid_3d_nn_local);
  cwork7 = fftw_alloc_complex(grid_3d_nn_local);

  for(idx = 0; idx < grid_3d_nn_local*2; idx++) {
    rwork1[idx] = 0.0;
    rwork2[idx] = 0.0;
    rwork3[idx] = 0.0;
    rwork4[idx] = 0.0;
    rwork5[idx] = 0.0;
    rwork6[idx] = 0.0;
    rwork7[idx] = 0.0;
  }

  for(idx = 0; idx < grid_3d_nn_local; idx++) {
    cwork1[idx] = 0.0 + 0.0*I;
    cwork2[idx] = 0.0 + 0.0*I;
    cwork3[idx] = 0.0 + 0.0*I;
    cwork4[idx] = 0.0 + 0.0*I;
    cwork5[idx] = 0.0 + 0.0*I;
    cwork6[idx] = 0.0 + 0.0*I;
    cwork7[idx] = 0.0 + 0.0*I;
  }

  // remove divergence from input velocity field
  ins3d_p_adjust(kq);

  ins3d_sigma = grid_dz*2.0;

  return 0;
}

int ins3d_finalize() {
  fftw_free(rwork1);
  fftw_free(rwork2);
  fftw_free(rwork3);
  fftw_free(rwork4);
  fftw_free(rwork5);
  fftw_free(rwork6);
  fftw_free(rwork7);

  fftw_free(cwork1);
  fftw_free(cwork2);
  fftw_free(cwork3);
  fftw_free(cwork4);
  fftw_free(cwork5);
  fftw_free(cwork6);
  fftw_free(cwork7);

  return 0;
}

// nonlinear part -- careful, careful!
void ins3d_adv(fluid_complex *kadv, fluid_complex *kstate) {
  ptrdiff_t idx2d, idx3d, m, n;

  fluid_real kx, ky, kz;

  int kk;

  // get the velocity in physical space
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      // dealias
      if(grid_2d_dealias_mask[idx2d] || 3*abs(kk) >= grid_nz) {
        cwork4[idx3d] = 0.0 + 0.0*I;
        cwork5[idx3d] = 0.0 + 0.0*I;
        cwork6[idx3d] = 0.0 + 0.0*I;
      }
      else {
        cwork4[idx3d] =  kstate[idx3d + grid_3d_nn_local*0];
        cwork5[idx3d] =  kstate[idx3d + grid_3d_nn_local*1];
        cwork6[idx3d] =  kstate[idx3d + grid_3d_nn_local*2];
      }
    }
  }

  spectral2physical(cwork4,rwork4, GRID_VERTICAL_LAYOUT_PERIODIC);
  spectral2physical(cwork5,rwork5, GRID_VERTICAL_LAYOUT_PERIODIC);
  spectral2physical(cwork6,rwork6, GRID_VERTICAL_LAYOUT_PERIODIC);

  // calculate the advection of non-velocity components
  for(n = 3; n < nq; n++) {
    for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
      kx = grid_2d_kx[idx2d];
      ky = grid_2d_ky[idx2d];
      for(m = 0; m < grid_nz; m++) {
        idx3d = idx2d + grid_2d_nn_local*m;
        kk = (2*m < grid_nz) ? m : m-grid_nz;
        kz = 2.0*M_PI*kk/grid_lz;

        // dealias
        if(grid_2d_dealias_mask[idx2d] || 3*abs(kk) >= grid_nz) {
          cwork1[idx3d] = 0.0 + 0.0*I;
          cwork2[idx3d] = 0.0 + 0.0*I;
          cwork3[idx3d] = 0.0 + 0.0*I;
        }
        else {
          cwork1[idx3d] = I*kx*kstate[idx3d + grid_3d_nn_local*n];
          cwork2[idx3d] = I*ky*kstate[idx3d + grid_3d_nn_local*n];
          cwork3[idx3d] = I*kz*kstate[idx3d + grid_3d_nn_local*n];
        }
      }
    }

    spectral2physical(cwork1, rwork1, GRID_VERTICAL_LAYOUT_PERIODIC);
    spectral2physical(cwork2, rwork2, GRID_VERTICAL_LAYOUT_PERIODIC);
    spectral2physical(cwork3, rwork3, GRID_VERTICAL_LAYOUT_PERIODIC);

    for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
      rwork1[idx3d] = rwork1[idx3d]*rwork4[idx3d]
                    + rwork2[idx3d]*rwork5[idx3d]
                    + rwork3[idx3d]*rwork6[idx3d];
    }

    physical2spectral(rwork1, &kadv[grid_3d_nn_local*n], GRID_VERTICAL_LAYOUT_PERIODIC);
  }

  // calculate the vorticity
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    kx = grid_2d_kx[idx2d];
    ky = grid_2d_ky[idx2d];
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      kz = 2.0*M_PI*kk/grid_lz;

      // dealias
      if(grid_2d_dealias_mask[idx2d] || 3*abs(kk) >= grid_nz) {
        cwork1[idx3d] = 0.0 + 0.0*I;
        cwork2[idx3d] = 0.0 + 0.0*I;
        cwork3[idx3d] = 0.0 + 0.0*I;
      }
      else {
        cwork1[idx3d] = kstate[idx3d + grid_3d_nn_local*2]*I*ky
                      - kstate[idx3d + grid_3d_nn_local*1]*I*kz;
        cwork2[idx3d] = kstate[idx3d + grid_3d_nn_local*0]*I*kz
                      - kstate[idx3d + grid_3d_nn_local*2]*I*kx;
        cwork3[idx3d] = kstate[idx3d + grid_3d_nn_local*1]*I*kx
                      - kstate[idx3d + grid_3d_nn_local*0]*I*ky;
      }
    }
  }
  spectral2physical(cwork1, rwork1, GRID_VERTICAL_LAYOUT_PERIODIC);
  spectral2physical(cwork2, rwork2, GRID_VERTICAL_LAYOUT_PERIODIC);
  spectral2physical(cwork3, rwork3, GRID_VERTICAL_LAYOUT_PERIODIC);

  // calculate vorticity x velocity
  // this is equal to the velocity advection
  // up to a curl-free factor which is removed
  // by the pressure correction
  for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
    rwork7[idx3d] = rwork2[idx3d]*rwork6[idx3d]
                  - rwork3[idx3d]*rwork5[idx3d];
  }
  physical2spectral(rwork7, &kadv[grid_3d_nn_local*0],
                    GRID_VERTICAL_LAYOUT_PERIODIC);
  for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
    rwork7[idx3d] = rwork3[idx3d]*rwork4[idx3d]
                  - rwork1[idx3d]*rwork6[idx3d];
  }
  physical2spectral(rwork7, &kadv[grid_3d_nn_local*1],
                    GRID_VERTICAL_LAYOUT_PERIODIC);
  for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
    rwork7[idx3d] = rwork1[idx3d]*rwork5[idx3d]
                  - rwork2[idx3d]*rwork4[idx3d];
  }
  physical2spectral(rwork7, &kadv[grid_3d_nn_local*2],
                    GRID_VERTICAL_LAYOUT_PERIODIC);
  
}

void ins3d_p_adjust(fluid_complex *kstate) {
  ptrdiff_t idx2d, idx3d, m;

  int kk;

  fluid_real kx, ky, kz;

  fluid_complex kp;

  // adjust velocity based on pressure forcing
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      if(!grid_2d_dealias_mask[idx2d] && 3*abs(kk) < grid_nz) {
        idx3d = idx2d + grid_2d_nn_local*m;
        kx = grid_2d_kx[idx2d];
        ky = grid_2d_ky[idx2d];
        kz = 2.0*M_PI*kk/grid_lz;
        kp = 0.0 + 0.0*I;
        if(fabs(kx) > 1e-14 ||
           fabs(ky) > 1e-14 ||
           fabs(kz) > 1e-14) {
          kp = kstate[grid_3d_nn_local*0 + idx3d]*kx*I
             + kstate[grid_3d_nn_local*1 + idx3d]*ky*I
             + kstate[grid_3d_nn_local*2 + idx3d]*kz*I;
          kp/= -kx*kx - ky*ky - kz*kz;
        }

        kstate[grid_3d_nn_local*0 + idx3d] -= I*kx*kp;
        kstate[grid_3d_nn_local*1 + idx3d] -= I*ky*kp;
        kstate[grid_3d_nn_local*2 + idx3d] -= I*kz*kp;

      }
    }
  }
}

void ins3d_rhs(fluid_complex *krhs, fluid_complex *kstate) {
  ptrdiff_t idx2d,idx3d,m,n;

  fluid_complex ku, kv, kw, kb;

  int kk;

  // calculate the advection
  ins3d_adv(krhs, kstate);

  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      for(n = 0; n < nq; n++) {
        if(grid_2d_dealias_mask[idx2d] || abs(kk)*3 >= grid_nz) {
          krhs[grid_3d_nn_local*n + idx3d] = 0.0;
        }
        else {
          if(cabs(krhs[grid_3d_nn_local*n + idx3d]) < 1e-14) {
            krhs[grid_3d_nn_local*n + idx3d] = 0.0;
          }
          else {
            krhs[grid_3d_nn_local*n + idx3d] *= -1.0;
            //krhs[grid_3d_nn_local*n + idx3d] = 0.0;
          }
        }
      }
    }
  }

  // apply coriolis force
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    ku = kstate[grid_3d_nn_local*0 + idx3d];
    kv = kstate[grid_3d_nn_local*1 + idx3d];
    if(cabs(ku) > 1e-14) {
      krhs[grid_3d_nn_local*1 + idx3d] -= ku*ins3d_fcor;
    }
    if(cabs(kv) > 1e-14) {
      krhs[grid_3d_nn_local*0 + idx3d] += kv*ins3d_fcor;
    }
  }

  // bouyancy forcing
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    kw = kstate[grid_3d_nn_local*2 + idx3d];
    kb = kstate[grid_3d_nn_local*3 + idx3d];
    if(cabs(kb) > 1e-14) {
      krhs[grid_3d_nn_local*2 + idx3d] += kb*ins3d_bfreq;
    }
    if(cabs(kw) > 1e-14) {
      krhs[grid_3d_nn_local*3 + idx3d] -= kw*ins3d_bfreq;
    }
  }

  // adjust velocity based on pressure forcing
  ins3d_p_adjust(krhs);
}

// Cash-Karp method of adaptive rk4
fluid_real ins3d_step_rk4_adaptive(fluid_real dt, fluid_real err_bnd_global) {
  ptrdiff_t ai, bi, n, idx2d, idx3d, m;

  fluid_complex *ks, *krhs;

  int kk;

  fluid_real ksq_2d, ksq_vd;

  fluid_real err_max = 0.0,
         err, err_max_global;

  fluid_complex kerr, knorm;

  fluid_real norm;

  const int runge_kutta_num = 6;

  const fluid_real a[5][5] =
  {{1./5.       , 0.0      , 0.0        , 0.0           , 0.0       },
   {3./40.      , 9./40.   , 0.0        , 0.0           , 0.0       },
   {3./10.      , -9./10.  , 6./5.      , 0.0           , 0.0       },
   {-11./54.    , 5./2.    , -70./27.   , 35./27.       , 0.0       },
   {1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.}};
 
  //const fluid_real c[5] = {1./5.,3./10., 3./5., 1., 7./8.};

  const fluid_real b[6] = {37./378.,  0., 250./621.,
                       125./594., 0., 512./1771.};

  const fluid_real d[6] = {2825./27648., 0.,
                       18575./48384.,
                       13525./55296.,
                       277/14336., 0.25};

  ks   = fftw_alloc_complex(grid_3d_nn_local*nq*runge_kutta_num);
  krhs = fftw_alloc_complex(grid_3d_nn_local*nq);

  // calculate the runge kutta coefficients (ks)
  for(bi = 0; bi < runge_kutta_num; bi++) {
    for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
      for(n = 0; n < nq; n++) {
        ks[grid_3d_nn_local*(nq*bi + n) + idx3d] = kq[grid_3d_nn_local*n + idx3d];

        for(ai = 1; ai <= bi; ai++) {
          ks[grid_3d_nn_local*(nq*bi + n) + idx3d] +=
            dt*a[bi-1][ai-1]*ks[grid_3d_nn_local*(nq*(ai-1) + n) + idx3d];
        }
      }
    }

    // calculate rhs
    ins3d_rhs(krhs, &ks[grid_3d_nn_local*nq*bi]);
    for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
      for(n = 0; n < nq; n++) {
        ks[grid_3d_nn_local*(nq*bi + n) + idx3d] = krhs[grid_3d_nn_local*n + idx3d];
      }
    }

  }  // end of loop over bi

  // calculate the maximum error
  err_max = 0.0;
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      // dealias
      if(!grid_2d_dealias_mask[idx2d] && 3*abs(kk) < grid_nz) {
        for(n = 0; n < 3; n++) {
          kerr = 0.0;
          knorm = 0.0;
          for(bi = 0; bi < runge_kutta_num; bi++) {
            kerr += dt*(b[bi]-d[bi])*ks[grid_3d_nn_local*(bi*nq + n) + idx3d];
            knorm += dt*b[bi]*ks[grid_3d_nn_local*(bi*nq + n) + idx3d];
          }
          norm = cabs(knorm);
          norm = (norm < 1e-12) ? 1e-12 : norm;
          err = cabs(kerr)/norm;
          if (err > err_max) err_max = err;
        }
      }
    }
  }

  MPI_Reduce(&err_max,  &err_max_global,  1,
             MPI_DOUBLE, MPI_MAX, master_task, MPI_COMM_WORLD);

  MPI_Bcast(&err_max_global, 1, MPI_DOUBLE,
            master_task, MPI_COMM_WORLD);

  if (err_max_global > err_bnd_global) {
    fftw_free(ks);
    fftw_free(krhs);
    return err_max_global;
  }

  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    for(bi = 0; bi < runge_kutta_num; bi++) {
      for(n = 0; n < nq; n++) {
        if(cabs(ks[grid_3d_nn_local*(bi*nq + n) + idx3d]) > 1e-14) {
          kq[grid_3d_nn_local*n + idx3d]
            += dt*b[bi]*ks[grid_3d_nn_local*(bi*nq + n) + idx3d];
        }
      }
    }
  }

  // apply viscous damping
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    ksq_2d = grid_2d_kx[idx2d]*grid_2d_kx[idx2d]
           + grid_2d_ky[idx2d]*grid_2d_ky[idx2d];
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      ksq_vd = pow(2.0*M_PI*kk,2);

      for(n = 0; n < 3; n++) {
        kq[grid_3d_nn_local*n + idx3d] *= exp(-pow(ksq_2d/(0.9*grid_2d_ksq_max),4)
                                              -pow(ksq_vd/(0.9*grid_vd_ksq_max),4));
      }
    }
  }

  // dealias kq
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      kk = (2*m < grid_nz) ? m : m-grid_nz;
      if(grid_2d_dealias_mask[idx2d] || 3*abs(kk) >= grid_nz) {
        for(n = 0; n < nq; n++) {
          kq[grid_3d_nn_local*n + idx3d] = 0.0;
        }
      }
    }
  }

  fftw_free(ks);
  fftw_free(krhs);
  return err_max_global;
}


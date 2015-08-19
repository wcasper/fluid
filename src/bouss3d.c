#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <iniparser.h>

#include "bouss3d.h"
#include "fluid.h"
#include "grid.h"
#include "comm.h"
#include "error.h"
#include "topo.h"
#include "fourier.h"
#include "state.h"
#include "config.h"

fluid_real bouss3d_kvisc    = 0.0;	// kinematic viscosity
fluid_real bouss3d_fcor     = 0.0;	// coriolis parameter
fluid_real bouss3d_sigma;		// boundary relax scale
fluid_real *bouss3d_b_freq;		// bouyancy frequency

fluid_real *rwork1, *rwork2, *rwork3,
           *rwork4, *rwork5, *rwork6;

fluid_complex *cwork1, *cwork2, *cwork3,
              *cwork4, *cwork5, *cwork6; 

static void bouss3d_adv(fluid_complex *kadv, fluid_complex *kstate);
static void bouss3d_rhs(fluid_complex *krhs, fluid_complex *kstate);
static void bouss3d_add_bouyancy_rhs(fluid_complex *krhs, fluid_complex *kstate);
static void bouss3d_p_adjust(fluid_complex *kstate);
static void bouss3d_topo_f(fluid_complex *kstate, fluid_real dt);
static void bouss3d_noslip_bottom(fluid_real *work, fluid_real dt);

static int bouss3d_read_config();

int bouss3d_read_config() {
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
    // read in bouss3d initialization data
    bouss3d_fcor  = iniparser_getdouble(dict, "bouss3d:fcor",  bouss3d_fcor);
    bouss3d_kvisc = iniparser_getdouble(dict, "bouss3d:kvisc", bouss3d_kvisc);
    iniparser_freedict(dict);
  }

  MPI_Bcast(&bouss3d_fcor,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&bouss3d_kvisc,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);

  return status;
}

int bouss3d_init() {
  int idx;

  int status = 0;

  fluid_real bf;

  // sanity check
  status = (nq != 4);
  error_check(&status, "wrong number of physical variables\n");
  if(status) return status;

  // sanity check
  status = (grid_nd != 3);
  error_check(&status, "wrong number of dimensions\n");
  if(status) return status;

  // read in config file
  status = bouss3d_read_config();
  error_check(&status, "error in bouss3d_read_config\n");
  if(status) return status;

  // initialize the bouyancy frequency
  bouss3d_b_freq = calloc(grid_nz,sizeof(fluid_real));
  for(idx = 0; idx < grid_nz; idx++) {
    bf = grid_vd_z[idx];
    bf = 0.005*exp(-bf*bf);
    bf = 0.001;
    bouss3d_b_freq[idx] = bf;
  }

  // initialize work arrays
  rwork1 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork2 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork3 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork4 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork5 = fftw_alloc_real(grid_3d_nn_local*2);
  rwork6 = fftw_alloc_real(grid_3d_nn_local*2);

  cwork1 = fftw_alloc_complex(grid_3d_nn_local);
  cwork2 = fftw_alloc_complex(grid_3d_nn_local);
  cwork3 = fftw_alloc_complex(grid_3d_nn_local);
  cwork4 = fftw_alloc_complex(grid_3d_nn_local);
  cwork5 = fftw_alloc_complex(grid_3d_nn_local);
  cwork6 = fftw_alloc_complex(grid_3d_nn_local);

  for(idx = 0; idx < grid_3d_nn_local*2; idx++) {
    rwork1[idx] = 0.0;
    rwork2[idx] = 0.0;
    rwork3[idx] = 0.0;
    rwork4[idx] = 0.0;
    rwork5[idx] = 0.0;
    rwork6[idx] = 0.0;
  }

  for(idx = 0; idx < grid_3d_nn_local; idx++) {
    cwork1[idx] = 0.0 + 0.0*I;
    cwork2[idx] = 0.0 + 0.0*I;
    cwork3[idx] = 0.0 + 0.0*I;
    cwork4[idx] = 0.0 + 0.0*I;
    cwork5[idx] = 0.0 + 0.0*I;
    cwork6[idx] = 0.0 + 0.0*I;
  }

  // initialize topography
  topo_init();

  bouss3d_sigma = grid_dz*2.0;
//  bouss3d_topo_f(kq, 1.0);

  return 0;
}

int bouss3d_finalize() {
  free(bouss3d_b_freq);

  fftw_free(rwork1);
  fftw_free(rwork2);
  fftw_free(rwork3);
  fftw_free(rwork4);
  fftw_free(rwork5);
  fftw_free(rwork6);

  fftw_free(cwork1);
  fftw_free(cwork2);
  fftw_free(cwork3);
  fftw_free(cwork4);
  fftw_free(cwork5);
  fftw_free(cwork6);

  topo_finalize();

  return 0;
}

// nonlinear part -- careful, careful!
void bouss3d_adv(fluid_complex *kadv, fluid_complex *kstate) {
  ptrdiff_t idx2d, idx3d, m, n;

  fluid_real kx, ky, kz;

  // get the velocity in physical space
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      // dealias
      if(grid_2d_dealias_mask[idx2d] || 3*m >= 2*grid_nz) {
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

  spectral2physical(cwork4,rwork4, state_layout[0]);
  spectral2physical(cwork5,rwork5, state_layout[1]);
  spectral2physical(cwork6,rwork6, state_layout[2]);

  // calculate the advection
  for(n = 0; n < nq; n++) {
    for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
      kx = grid_2d_kx[idx2d];
      ky = grid_2d_ky[idx2d];
      for(m = 0; m < grid_nz; m++) {
        idx3d = idx2d + grid_2d_nn_local*m;

        // dealias
        if(grid_2d_dealias_mask[idx2d] || 3*m >= 2*grid_nz) {
          cwork1[idx3d] = 0.0 + 0.0*I;
          cwork2[idx3d] = 0.0 + 0.0*I;
          cwork3[idx3d] = 0.0 + 0.0*I;
        }
        else {
          cwork1[idx3d] = I*kx*kstate[idx3d + grid_3d_nn_local*n];
          cwork2[idx3d] = I*ky*kstate[idx3d + grid_3d_nn_local*n];
          if(state_layout[n] == GRID_VERTICAL_LAYOUT_COSINE) {
            //shift down
            if(m >= grid_nz-1) {
              cwork3[idx3d] = 0.0;
            }
            else {
              kz = -M_PI*(m+1)/grid_lz;
              cwork3[idx3d] =  kz*kstate[idx3d + grid_2d_nn_local + grid_3d_nn_local*n];
            }
          }
          else {
            //shift up
            if(m == 0) {
              cwork3[idx3d] = 0.0;
            }
            else {
              kz = M_PI*m/grid_lz;
              cwork3[idx3d] =  kz*kstate[idx3d - grid_2d_nn_local + grid_3d_nn_local*n];
            }
          }
        }
      }
    }

    if(state_layout[n] == GRID_VERTICAL_LAYOUT_COSINE) {
      spectral2physical(cwork1, rwork1, GRID_VERTICAL_LAYOUT_COSINE);
      spectral2physical(cwork2, rwork2, GRID_VERTICAL_LAYOUT_COSINE);
      spectral2physical(cwork3, rwork3, GRID_VERTICAL_LAYOUT_SINE);
    }
    else {
      spectral2physical(cwork1, rwork1, GRID_VERTICAL_LAYOUT_SINE);
      spectral2physical(cwork2, rwork2, GRID_VERTICAL_LAYOUT_SINE);
      spectral2physical(cwork3, rwork3, GRID_VERTICAL_LAYOUT_COSINE);
    }

    for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
      rwork1[idx3d] = rwork1[idx3d]*rwork4[idx3d]
                    + rwork2[idx3d]*rwork5[idx3d]
                    + rwork3[idx3d]*rwork6[idx3d];
    }

    if(state_layout[n] == GRID_VERTICAL_LAYOUT_COSINE) {
      physical2spectral(rwork1, &kadv[grid_3d_nn_local*n], GRID_VERTICAL_LAYOUT_COSINE);
    }
    else {
      physical2spectral(rwork1, &kadv[grid_3d_nn_local*n], GRID_VERTICAL_LAYOUT_SINE);
    }
  }
}

void bouss3d_p_adjust(fluid_complex *kstate) {
  ptrdiff_t idx2d, idx3d, m;

  fluid_real kx, ky, kz;

  fluid_complex kp;

  // adjust velocity based on pressure forcing
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
        idx3d = idx2d + grid_2d_nn_local*m;
        kx = grid_2d_kx[idx2d];
        ky = grid_2d_ky[idx2d];
        kz = M_PI*m/grid_lz;
        kp = 0.0 + 0.0*I;
        if(fabs(kx) > 1e-14 ||
           fabs(ky) > 1e-14 || m > 0) {
          kp = kstate[grid_3d_nn_local*0 + idx3d]*kx*I
             + kstate[grid_3d_nn_local*1 + idx3d]*ky*I;
          if(m > 0) {
            kp += kstate[grid_3d_nn_local*2 - grid_2d_nn_local + idx3d]*kz;
            kp/= -kx*kx - ky*ky - kz*kz;
          }
          else {
            kp/= -kx*kx - ky*ky;
          }
        }

        kstate[grid_3d_nn_local*0 + idx3d] -= I*kx*kp;
        kstate[grid_3d_nn_local*1 + idx3d] -= I*ky*kp;
        if(m > 0) {
          kstate[grid_3d_nn_local*2 - grid_2d_nn_local + idx3d] -= -kz*kp;
        }
    }
  }
}

void bouss3d_rhs(fluid_complex *krhs, fluid_complex *kstate) {
  ptrdiff_t idx2d,idx3d,m,n;

  fluid_complex ku, kv, kw, kb;

  // calculate the advection
  //bouss3d_adv(krhs, kstate);

  /*
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    kq[grid_3d_nn_local*0 + idx3d] = krhs[grid_3d_nn_local*0 + idx3d];
    kq[grid_3d_nn_local*1 + idx3d] = krhs[grid_3d_nn_local*1 + idx3d];
    kq[grid_3d_nn_local*2 + idx3d] = krhs[grid_3d_nn_local*2 + idx3d];
    kq[grid_3d_nn_local*3 + idx3d] = krhs[grid_3d_nn_local*3 + idx3d];
  }
  state_write("advection.bin");
  exit(EXIT_FAILURE);
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    krhs[grid_3d_nn_local*0 + idx3d] = 0.0;
    krhs[grid_3d_nn_local*1 + idx3d] = 0.0;
    krhs[grid_3d_nn_local*2 + idx3d] = 0.0;
    krhs[grid_3d_nn_local*3 + idx3d] = 0.0;
  }

  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      for(n = 0; n < nq; n++) {
            krhs[grid_3d_nn_local*n + idx3d] *= -1.0;
      }
    }
  }
  */

  // apply coriolis force
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    ku = kstate[grid_3d_nn_local*0 + idx3d];
    kv = kstate[grid_3d_nn_local*1 + idx3d];
      krhs[grid_3d_nn_local*1 + idx3d] -= ku*bouss3d_fcor;
      krhs[grid_3d_nn_local*0 + idx3d] += kv*bouss3d_fcor;
  }

  // apply bouyancy rhs forcing
  // affects wvel and bouy terms
  //bouss3d_add_bouyancy_rhs(krhs,kstate);
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    kw = kstate[grid_3d_nn_local*2 + idx3d];
    kb = kstate[grid_3d_nn_local*3 + idx3d];
      krhs[grid_3d_nn_local*2 + idx3d] += kb;
      krhs[grid_3d_nn_local*3 + idx3d] -= kw*1e-6;
  }

  // adjust velocity based on pressure forcing
  bouss3d_p_adjust(krhs);
}

void bouss3d_noslip_bottom(fluid_real *work, fluid_real dt) {
  int idx2d, m, idx3d, topo_idx;

  fluid_real coeff;

  for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*2*m;
      topo_idx = grid_nz*(grid_ny*grid_2d_i[idx2d]+ grid_2d_j[idx2d]) + m;
      coeff = topo_d2b[topo_idx];
      coeff /= bouss3d_sigma;
      coeff = coeff*coeff;
      //coeff = 1.0-exp(-coeff*coeff);
      if(coeff < 1e-16) {
        coeff = 0.0;
      }
      else {
        coeff = exp(-dt/coeff);
      }
      work[idx3d] *= coeff;
    }
  }
}

void bouss3d_topo_f(fluid_complex *kstate, fluid_real dt) {
  ptrdiff_t idx3d;

  // get the velocity in physical space
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    cwork4[idx3d] =  kstate[idx3d + grid_3d_nn_local*0];
    cwork5[idx3d] =  kstate[idx3d + grid_3d_nn_local*1];
    cwork6[idx3d] =  kstate[idx3d + grid_3d_nn_local*2];
  }

  spectral2physical(cwork4,rwork4, GRID_VERTICAL_LAYOUT_COSINE);
  spectral2physical(cwork5,rwork5, GRID_VERTICAL_LAYOUT_COSINE);
  spectral2physical(cwork6,rwork6, GRID_VERTICAL_LAYOUT_SINE);

  // velocity forced toward zero on topography
  // and at the fluid surface
  bouss3d_noslip_bottom(rwork4, dt);
  bouss3d_noslip_bottom(rwork5, dt);
  bouss3d_noslip_bottom(rwork6, dt);

  physical2spectral(rwork4,cwork4, GRID_VERTICAL_LAYOUT_COSINE);
  physical2spectral(rwork5,cwork5, GRID_VERTICAL_LAYOUT_COSINE);
  physical2spectral(rwork6,cwork6, GRID_VERTICAL_LAYOUT_SINE);

  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    kstate[idx3d + grid_3d_nn_local*0] = cwork4[idx3d];
    kstate[idx3d + grid_3d_nn_local*1] = cwork5[idx3d];
    kstate[idx3d + grid_3d_nn_local*2] = cwork6[idx3d];
  }

  // zero the divergence
  bouss3d_p_adjust(kstate);
}

void bouss3d_add_bouyancy_rhs(fluid_complex *krhs, fluid_complex *kstate) {
  ptrdiff_t idx3d,idx2d,m;

  // get the vertical velocity in physical space
  memcpy(cwork6,&kstate[grid_3d_nn_local*2],grid_3d_nn_local*sizeof(fluid_complex));
  spectral2physical(cwork6,rwork6, GRID_VERTICAL_LAYOUT_SINE);

  // multiply w and b_freq
  for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*2*m;
      rwork6[idx3d] = -rwork6[idx3d]*pow(bouss3d_b_freq[m],2);
    }
  }
  physical2spectral(rwork6,cwork6, GRID_VERTICAL_LAYOUT_SINE);
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    krhs[idx3d + grid_3d_nn_local*3] += cwork6[idx3d];
    krhs[idx3d + grid_3d_nn_local*2] += kstate[idx3d + grid_3d_nn_local*3];
  }
}

// Cash-Karp method of adaptive rk4
fluid_real bouss3d_step_rk4_adaptive(fluid_real dt, fluid_real err_bnd_global) {
  ptrdiff_t ai, bi, n, idx2d, idx3d, m;

  fluid_complex *ks, *krhs;

  fluid_real ksq_2d, ksq_vd;

  fluid_real err_max = 0.0,
         err, err_max_global;

  fluid_complex kerr, knorm;

  fluid_real norm;

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

  const int runge_kutta_num = 6;

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
    bouss3d_rhs(krhs, &ks[grid_3d_nn_local*nq*bi]);
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
      // dealias
      if(!grid_2d_dealias_mask[idx2d] && 3*m < 2*grid_nz) {
        for(n = 0; n < 3; n++) {
          kerr = 0.0;
          knorm = 0.0;
          for(bi = 0; bi < runge_kutta_num; bi++) {
            kerr += dt*(b[bi]-d[bi])*ks[grid_3d_nn_local*(bi*nq + n) + idx3d];
            knorm += dt*b[bi]*ks[grid_3d_nn_local*(bi*nq + n) + idx3d];
          }
          norm = cabs(knorm);
          norm = (norm < 1e-8) ? 1e-8 : norm;
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
        kq[grid_3d_nn_local*n + idx3d]
          += dt*b[bi]*ks[grid_3d_nn_local*(bi*nq + n) + idx3d];
      }
    }
  }
  fftw_free(ks);
  fftw_free(krhs);

  // apply viscous damping
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    ksq_2d = grid_2d_kx[idx2d]*grid_2d_kx[idx2d]
           + grid_2d_ky[idx2d]*grid_2d_ky[idx2d];
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;


      for(n = 0; n < 3; n++) {
        if(state_layout[n] == GRID_VERTICAL_LAYOUT_COSINE) {
          ksq_vd = grid_vd_kzo[m]*grid_vd_kzo[m];
//          kq[grid_3d_nn_local*n + idx3d] *= exp(-pow(ksq_2d/(0.9*grid_2d_ksq_max),4)
//                                                -pow(ksq_vd/(0.9*grid_vd_ksq_max),4));
          kq[grid_3d_nn_local*n + idx3d] /= 1.0 + (ksq_2d+ksq_vd)*dt*bouss3d_kvisc;
        }
        else {
          ksq_vd = grid_vd_kze[m]*grid_vd_kze[m];
//          kq[grid_3d_nn_local*n + idx3d] *= exp(-pow(ksq_2d/(0.9*grid_2d_ksq_max),4)
//                                                -pow(ksq_vd/(0.9*grid_vd_ksq_max),4));
          kq[grid_3d_nn_local*n + idx3d] /= 1.0 + (ksq_2d+ksq_vd)*dt*bouss3d_kvisc;
        }
      }
    }
  }

  // apply boundary adjustment
  //bouss3d_topo_f(kq, dt);

  // dealias kq
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      if(grid_2d_dealias_mask[idx2d] || 3*m >= 2*grid_nz) {
        for(n = 0; n < nq; n++) {
          kq[grid_3d_nn_local*n + idx3d] = 0.0;
        }
      }
    }
  }

  return err_max_global;
}


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <assert.h>
#include <iniparser.h>

#include "state.h"
#include "fluid.h"
#include "grid.h"
#include "comm.h"
#include "model.h"
#include "fourier.h"
#include "config.h"
#include "error.h"
#include "diag.h"

fluid_real *q;
fluid_complex *kq;
int nq = 0;
grid_vertical_layout_t *state_layout;

fluid_complex *state_kvort;
fluid_real *state_vort;

fluid_real *rwork;
fluid_complex *cwork;

FILE *state_ifile,
     *state_ofile;

state_init_type_t state_init_type = STATE_INIT_TYPE_PATCHES_2D;

char * state_restart_file_name = NULL;

static int state_read_config();

int state_init() {
  int idx2d, idx3d, m, n;

  int status = 0;

  fluid_real x, y, z;

  status = state_read_config();
  error_check(&status, "error in state_read_config\n");
  if(status) return status;

  q  = fftw_alloc_real(grid_3d_nn_local*2*nq);
  kq = fftw_alloc_complex(grid_3d_nn_local*nq);
  state_layout = calloc(nq, sizeof(grid_vertical_layout_t));

  state_vort = fftw_alloc_real(grid_3d_nn_local*2);
  state_kvort = fftw_alloc_complex(grid_3d_nn_local);

  rwork = fftw_alloc_real(grid_3d_nn_local*2*nq);
  cwork = fftw_alloc_complex(grid_3d_nn_local*nq);

  // zero everything
  for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
    for(n = 0; n < nq; n++) {
      q[grid_3d_nn_local*2*n + idx3d] = 0.0;
    }
    state_vort[idx3d] = 0.0;
  }
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    for(n = 0; n < nq; n++) {
      kq[grid_3d_nn_local*n + idx3d] = 0.0 + 0.0*I;
    }
    state_kvort[idx3d] = 0.0 + 0.0*I;
  }

  // initialize vertical profiles
  switch(model_type) {
    case MODEL_INS2D:
      state_layout[0] = GRID_VERTICAL_LAYOUT_PERIODIC;
      break;

    case MODEL_INS3D:
      state_layout[0] = GRID_VERTICAL_LAYOUT_PERIODIC;
      state_layout[1] = GRID_VERTICAL_LAYOUT_PERIODIC;
      state_layout[2] = GRID_VERTICAL_LAYOUT_PERIODIC;
      state_layout[3] = GRID_VERTICAL_LAYOUT_PERIODIC;
      break;

    case MODEL_BOUSS3D:
      state_layout[0] = GRID_VERTICAL_LAYOUT_COSINE;
      state_layout[1] = GRID_VERTICAL_LAYOUT_COSINE;
      state_layout[2] = GRID_VERTICAL_LAYOUT_SINE;
      state_layout[3] = GRID_VERTICAL_LAYOUT_SINE;
      break;

    default:
      status = 1;
      break;
  }
  error_check(&status, "unrecognized model type\n");
  if(status) return status;

  // Initialize spectral data
  double ke, pe;
  double fcor  = 0.0;
  double bfreq = 0.0;
  double umag= 0.0;
  double ro= 0.0;
  double fr= 0.0;
  dictionary *dict;
  switch(state_init_type) {
    case STATE_INIT_TYPE_RESTART:
      state_read(state_restart_file_name);
      state_physical2spectral();
      state_spectral2physical();
      state_physical2spectral();
      state_spectral2physical();
      state_physical2spectral();
      break;

    case STATE_INIT_TYPE_PATCHES_2D:
      for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
        x = grid_2d_x[idx2d];
        y = grid_2d_y[idx2d];
        q[idx2d]  = cos(2.0*M_PI*(x+y));
        q[idx2d] *= cos(2.0*M_PI*y);
        q[idx2d] *= 0.01;
      }
      state_physical2spectral();
      break;

    case STATE_INIT_TYPE_PATCHES_3D:
      for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
        for(m = 0; m < grid_nz; m++) {
          idx3d = idx2d + grid_2d_nn_local*2*m;
          x = grid_2d_x[idx2d];
          y = grid_2d_y[idx2d];
          q[idx3d + grid_3d_nn_local*2*0] = 0.01*cos(2.0*M_PI*y);
          q[idx3d + grid_3d_nn_local*2*1] = 0.01*cos(2.0*M_PI*x);
          q[idx3d + grid_3d_nn_local*2*2] = 0.01*cos(2.0*M_PI*(x+y));
        }
      }
      state_physical2spectral();
      break;

    case STATE_INIT_TYPE_BOUSS3D_TEST1:
      for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
        for(m = 0; m < grid_nz; m++) {
          x = grid_2d_x[idx2d];
          y = grid_2d_y[idx2d];
          z = grid_vd_z[m];
          idx3d = idx2d + grid_2d_nn_local*2*m;
          q[grid_3d_nn_local*2*0 + idx3d] = sin(4.0*M_PI*y/grid_ly)*cos(4.0*M_PI*z/grid_lz)*(grid_lz/grid_ly)*1.0e-1;
          q[grid_3d_nn_local*2*1 + idx3d] = sin(2.0*M_PI*x/grid_lx)*cos(2.0*M_PI*z/grid_lz)*(-grid_lz/grid_lx)*1.0e0;
          q[grid_3d_nn_local*2*3 + idx3d] = cos(2.0*M_PI*x/grid_lx)*sin(2.0*M_PI*z/grid_lz)*1.0e-4;
          q[grid_3d_nn_local*2*3 + idx3d]+= cos(4.0*M_PI*y/grid_ly)*sin(4.0*M_PI*z/grid_lz)*1.0e-5;
          //q[grid_3d_nn_local*2*0 + idx3d] = sin(2.0*M_PI*y/grid_ly)*cos(2.0*M_PI*z/grid_lz)*(grid_lz/grid_ly)*1.0e2;
          //q[grid_3d_nn_local*2*3 + idx3d] = 0.01*cos(2.0*M_PI*y/grid_ly)*sin(2.0*M_PI*z/grid_lz);
          //q[grid_3d_nn_local*2*1 + idx3d] = sin(2.0*M_PI*x/grid_lx)*cos(2.0*M_PI*z/grid_lz)*(-grid_lz/grid_lx)*1.0e2;
          //q[grid_3d_nn_local*2*3 + idx3d] = 0.01*cos(2.0*M_PI*x/grid_lx)*sin(2.0*M_PI*z/grid_lz);
          /*
          q[grid_3d_nn_local*2*0 + idx3d]*= 1e-1;
          q[grid_3d_nn_local*2*1 + idx3d]*= 1e-1;
          q[grid_3d_nn_local*2*2 + idx3d]*= 1e-1;
          q[grid_3d_nn_local*2*3 + idx3d]*= 1e-1;
          */
        }
      }
      write_energy(&ke, &pe);
      if(my_task == master_task) {
        printf("KE PE TE %1.16lf %1.16lf %1.16lf\n", ke, pe, ke+pe);
      }
      state_physical2spectral();
      break;

    case STATE_INIT_TYPE_INS3D_TEST1:
      if(my_task == master_task) {
        dict = iniparser_load(config_file_name);
        fcor  = iniparser_getdouble(dict, "ins3d:fcor", 0.0);
        bfreq = iniparser_getdouble(dict, "ins3d:bfreq", 0.0);
        umag = iniparser_getdouble(dict, "ins3d:umag", 0.0);
        iniparser_freedict(dict);
      }
      MPI_Bcast(&fcor,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
      MPI_Bcast(&bfreq,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
      MPI_Bcast(&umag,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
      ro = umag/(grid_lx*fcor);
      fr = umag/(bfreq*grid_lz);

      if(my_task == master_task) {
        printf("FCOR:  %1.16lf\n", fcor);
        printf("BFREQ: %1.16lf\n", bfreq);
        printf("UMAG:  %1.16lf\n", umag);
        printf("Ro:    %1.16lf\n", ro);
        printf("Fr:    %1.16lf\n", fr);
      }

      for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
        for(m = 0; m < grid_nz; m++) {
          x = grid_2d_x[idx2d];
          y = grid_2d_y[idx2d];
          z = m*grid_dz;
          idx3d = idx2d + grid_2d_nn_local*2*m;
          q[grid_3d_nn_local*2*0 + idx3d] = sin(4.0*M_PI*y/grid_ly)*cos(4.0*M_PI*z/grid_lz)*umag*(-1e-1);
          q[grid_3d_nn_local*2*0 + idx3d]+= sin(6.0*M_PI*y/grid_ly + .1)*cos(6.0*M_PI*z/grid_lz)*umag*(-1e-2);
          q[grid_3d_nn_local*2*1 + idx3d] = sin(2.0*M_PI*x/grid_lx)*cos(2.0*M_PI*z/grid_lz)*umag;
          //q[grid_3d_nn_local*2*1 + idx3d] = sin(2.0*M_PI*z/grid_lz);
          q[grid_3d_nn_local*2*3 + idx3d] = cos(2.0*M_PI*x/grid_lx)*sin(2.0*M_PI*z/grid_lz)*umag*(fr/ro);
          q[grid_3d_nn_local*2*3 + idx3d]+= cos(4.0*M_PI*y/grid_ly)*sin(4.0*M_PI*z/grid_lz)*umag*(fr/ro)*1e-1;
          q[grid_3d_nn_local*2*3 + idx3d]+= cos(6.0*M_PI*y/grid_ly + .1)*sin(6.0*M_PI*z/grid_lz)*umag*(fr/ro)*1e-2;
        }
      }
      state_physical2spectral();
      break;

    default:
      fprintf(stderr, "unknown state initialization\n");
      return 1;
  }

  return status;
}

int state_physical2spectral() {
  ptrdiff_t n,idx;

  //memcpy(rwork,q,grid_3d_nn_local*nq*2*sizeof(fluid_real));
  for(idx = 0; idx < grid_3d_nn_local*2*nq; idx++) {
    rwork[idx] = q[idx];
  }

  for(n = 0; n < nq; n++) {
    idx = grid_3d_nn_local*n;
    physical2spectral(&rwork[idx*2],&kq[idx],state_layout[n]);
  }

  return 0;
}

int state_spectral2physical() {
  ptrdiff_t n, idx;

  //memcpy(cwork,kq,grid_3d_nn_local*nq*sizeof(fluid_complex));
  for(idx = 0; idx < grid_3d_nn_local*nq; idx++) {
    cwork[idx] = kq[idx];
  }

  for(n = 0; n < nq; n++) {
    idx = grid_3d_nn_local*n;
    spectral2physical(&cwork[idx],&q[idx*2],state_layout[n]);
  }

  return 0;
}

int state_read(char *ifile_name) {
  int n, idx;

  FILE *ifile;

  fluid_real *q_global;

  size_t read_size = 0;

  read_size = grid_nx*(grid_ny/2 + 1)*2;
  if(grid_nd == 3) {
    read_size *= grid_nz;
  }

  if(my_task == master_task) {
    q_global  = calloc(read_size, sizeof(fluid_real));

    ifile = fopen(ifile_name, "rb");
    assert(ifile);
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_3d_nn_local*2;

    if(my_task == master_task) {
      fread(q_global, sizeof(fluid_real), read_size, ifile);
    }

    scatter_global_array(&q[idx], q_global, sizeof(fluid_real));
  }

  if(my_task == master_task) {
    fclose(ifile);
    free(q_global);
  }

  state_physical2spectral();

  return 0;
}

int state_write(char *ofile_name) {
  int n, idx;

  FILE *ofile;

  fluid_real *q_global;

  size_t write_size = 0;

  int idx2d, idx3d, m;
  fluid_complex foo;
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      for(n = 0; n < nq; n++) {
        foo = kq[idx3d + grid_3d_nn_local*n];
        if(cabs(foo) > 1e-12 && !grid_2d_dealias_mask[idx2d]) {
          //printf("hi!! %i %i %i %i ", grid_2d_ki[idx2d], grid_2d_kj[idx2d], m, n);
          //printf("%1.16lf %1.16lf\n", creal(foo), cimag(foo));
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  state_spectral2physical();
  double ke, pe;
  write_energy(&ke, &pe);
  if(my_task == master_task) {
    printf("KE PE TE %1.16lf %1.16lf %1.16lf\n", ke, pe, ke+pe);
  }

  write_size = grid_nx*(grid_ny/2 + 1)*2;
  if(grid_nd == 3) {
    write_size *= grid_nz;
  }

  if(my_task == master_task) {
    q_global = calloc(write_size, sizeof(fluid_real));
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name, "wb");
    assert(ofile);
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_3d_nn_local*2;

    gather_global_array(&q[idx], q_global, sizeof(fluid_real));

    if(my_task == master_task) {
      fwrite(q_global, sizeof(fluid_real), write_size, ofile);
    }
  }

  if(my_task == master_task) {
    fclose(ofile);
    free(q_global);
  }

  return 0;
}

int state_finalize() {
  fftw_free(q);
  fftw_free(kq);
  free(state_layout);
  fftw_free(rwork);
  fftw_free(cwork);

  fftw_free(state_vort);
  fftw_free(state_kvort);

  if(state_restart_file_name) free(state_restart_file_name);

  return 0;
}

int state_read_config() {
  int status = 0, len;

  dictionary *dict;

  char *file_name;

  int type;

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
    nq = iniparser_getint(dict, "state:nq", nq);
    state_init_type
      = (state_init_type_t)iniparser_getint(dict, "state:init", state_init_type);
    file_name = iniparser_getstring(dict, "state:rfile", NULL);
    len = 0;
    if(file_name) {
      len = strlen(file_name)+1;
      if(len > __FLUID_STRLEN_MAX) {
        status = 1;
      }
      else {
        state_restart_file_name = calloc(len,sizeof(char));
        strncpy(state_restart_file_name,file_name,len);
      }
    }
    type = iniparser_getint(dict, "model:type", model_type);
    iniparser_freedict(dict);
  }
  error_check(&status, "restart file name too long\n");
  if(status) return status;

  MPI_Bcast(&nq,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&state_init_type,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&type,1,MPI_INT,master_task,MPI_COMM_WORLD);
  model_type = (model_type_t) type;

  return status;
}

int state_write_vort(char *ofile_name) {
  int idx2d, idx3d, m;

  FILE *ofile;

  size_t write_size = 0;


  fluid_real *vort_global;

  if(grid_nd < 3) return 1;

  state_physical2spectral();
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;

      state_kvort[idx3d]  = kq[grid_3d_nn_local*1 + idx3d]*grid_2d_kx[idx2d]*I;
      state_kvort[idx3d] -= kq[grid_3d_nn_local*0 + idx3d]*grid_2d_ky[idx2d]*I;
       
    }
  }
  spectral2physical(state_kvort,state_vort,state_layout[0]);

  write_size = grid_nx*(grid_ny/2 + 1)*2;
  if(grid_nd == 3) {
    write_size *= grid_nz;
  }

  if(my_task == master_task) {
    vort_global = calloc(write_size, sizeof(fluid_real));
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name, "wb");
    assert(ofile);
  }

  gather_global_array(state_vort, vort_global, sizeof(fluid_real));

  if(my_task == master_task) {
    fwrite(vort_global, sizeof(fluid_real), write_size, ofile);
  }
  
  if(my_task == master_task) {
    fclose(ofile);
    free(vort_global);
  }

  return 0;
}





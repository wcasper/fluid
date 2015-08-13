#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <assert.h>
#include <iniparser.h>

#include "state.h"
#include "grid.h"
#include "comm.h"
#include "fourier.h"
#include "config.h"
#include "error.h"

double *q;
double complex *kq;
ptrdiff_t nq = 1;
grid_vertical_layout_t *state_layout;

double complex *state_kvort;
double *state_vort;

double *rwork;
double complex *cwork;

FILE *state_ifile,
     *state_ofile;

char ifile_name[256] = "init.ieeer8";

state_init_type_t state_init_type = STATE_INIT_TYPE_PATCHES_2D;

char * state_restart_file_name = NULL;

static int state_read_config();

int state_init() {
  int idx2d, idx3d, m, n;

  int status = 0;

  double x, y, z;

  status = state_read_config();
  error_check(&status, "error in state_read_config\n");
  if(status) return status;

  q  = fftw_alloc_real(grid_3d_nn_local*2*nq);
  kq = fftw_alloc_complex(grid_3d_nn_local*nq);
  state_layout = calloc(nq, sizeof(grid_vertical_layout_t));

  state_vort = fftw_alloc_real(grid_3d_nn_local);
  state_kvort = fftw_alloc_complex(grid_3d_nn_local);

  rwork = fftw_alloc_real(grid_3d_nn_local*2*nq);
  cwork = fftw_alloc_complex(grid_3d_nn_local*nq);

  // zero everything
  for(idx3d = 0; idx3d < grid_3d_nn_local*2; idx3d++) {
    for(n = 0; n < nq; n++) {
      q[grid_3d_nn_local*2*n + idx3d] = 0.0;
    }
  }
  for(idx3d = 0; idx3d < grid_3d_nn_local; idx3d++) {
    for(n = 0; n < nq; n++) {
      kq[grid_3d_nn_local*n + idx3d] = 0.0 + 0.0*I;
    }
  }

      MPI_Barrier(MPI_COMM_WORLD);
  // Initialize spectral data
  switch(state_init_type) {
    case STATE_INIT_TYPE_RESTART:
      state_read(state_restart_file_name);
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
      state_layout[0] = GRID_VERTICAL_LAYOUT_COSINE;
      state_layout[1] = GRID_VERTICAL_LAYOUT_COSINE;
      state_layout[2] = GRID_VERTICAL_LAYOUT_SINE;
      state_layout[3] = GRID_VERTICAL_LAYOUT_SINE;

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
      state_physical2spectral();
      break;

    case STATE_INIT_TYPE_INS3D_TEST1:
      state_layout[0] = GRID_VERTICAL_LAYOUT_PERIODIC;
      state_layout[1] = GRID_VERTICAL_LAYOUT_PERIODIC;
      state_layout[2] = GRID_VERTICAL_LAYOUT_PERIODIC;
      state_layout[3] = GRID_VERTICAL_LAYOUT_PERIODIC;

      for(idx2d = 0; idx2d < grid_2d_nn_local*2; idx2d++) {
        for(m = 0; m < grid_nz; m++) {
          x = grid_2d_x[idx2d];
          y = grid_2d_y[idx2d];
          z = m*grid_dz;
          idx3d = idx2d + grid_2d_nn_local*2*m;
          q[grid_3d_nn_local*2*0 + idx3d] = sin(4.0*M_PI*y/grid_ly)*cos(4.0*M_PI*z/grid_lz)*(grid_lz/grid_ly)*1e1;
          q[grid_3d_nn_local*2*1 + idx3d] = sin(2.0*M_PI*x/grid_lx)*cos(2.0*M_PI*z/grid_lz)*(-grid_lz/grid_lx)*1.0e2;
          q[grid_3d_nn_local*2*3 + idx3d] = cos(2.0*M_PI*x/grid_lx)*sin(2.0*M_PI*z/grid_lz)*1.0e-2;
          q[grid_3d_nn_local*2*3 + idx3d]+= 0.001*cos(4.0*M_PI*y/grid_ly)*sin(4.0*M_PI*z/grid_lz);
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

  //memcpy(rwork,q,grid_3d_nn_local*nq*2*sizeof(double));
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

  //memcpy(cwork,kq,grid_3d_nn_local*nq*sizeof(double complex));
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

  double *q_global;

  size_t read_size = 0;

  read_size = grid_nx*(grid_ny/2 + 1)*2;
  if(grid_nd == 3) {
    read_size *= grid_nz;
  }

  if(my_task == master_task) {
    q_global  = calloc(read_size, sizeof(double));

    ifile = fopen(ifile_name, "rb");
    assert(ifile);
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_3d_nn_local*2;

    if(my_task == master_task) {
      fread(q_global, sizeof(double), read_size, ifile);
    }

    scatter_global_array(&q[idx], q_global, sizeof(double));
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

  double *q_global;

  size_t write_size = 0;

  int idx2d, idx3d, m;
  double complex foo;
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + grid_2d_nn_local*m;
      for(n = 0; n < nq; n++) {
        foo = kq[idx3d + grid_3d_nn_local*n];
        if(cabs(foo) > 1e-12) {
          //printf("hi!! %i %i %i %i ", grid_2d_ki[idx2d], grid_2d_kj[idx2d], m, n);
          //printf("%1.16lf %1.16lf\n", creal(foo), cimag(foo));
        }
      }
    }
  }

  state_spectral2physical();

  write_size = grid_nx*(grid_ny/2 + 1)*2;
  if(grid_nd == 3) {
    write_size *= grid_nz;
  }

  if(my_task == master_task) {
    q_global = calloc(write_size, sizeof(double));
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name, "wb");
    assert(ofile);
  }

  for(n = 0; n < nq; n++) {
    idx = n*grid_3d_nn_local*2;

    gather_global_array(&q[idx], q_global, sizeof(double));

    if(my_task == master_task) {
      fwrite(q_global, sizeof(double), write_size, ofile);
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
      state_restart_file_name = calloc(len,sizeof(char));
      strcpy(state_restart_file_name,file_name);
    }
    iniparser_freedict(dict);
  }

  MPI_Bcast(&nq,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&state_init_type,1,MPI_INT,master_task,MPI_COMM_WORLD);

  return status;
}

int state_write_vort(char *ofile_name) {
  int idx2d, idx3d, m;

  FILE *ofile;

  size_t write_size = 0;


  double *vort_global;

  if(grid_nd < 2) return 1;

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
    vort_global = calloc(write_size, sizeof(double));
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name, "wb");
    assert(ofile);
  }

  gather_global_array(state_vort, vort_global, sizeof(double));

  if(my_task == master_task) {
    fwrite(vort_global, sizeof(double), write_size, ofile);
  }
  
  if(my_task == master_task) {
    fclose(ofile);
    free(vort_global);
  }

  return 0;
}





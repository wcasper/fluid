#include <complex.h>
#include <fftw3-mpi.h>
#include <assert.h>

#include "fluid.h"
#include "fourier.h"
#include "grid.h"

fluid_real *rwork2d;
fluid_real *rworkv;
fluid_complex *cwork2d;
fluid_complex *cworkv;

fftw_plan fp2d,		// two-dimensional forward plan
          rp2d,		// two-dimensional reverse plan
          fpvp,		// vertical forward periodic plan
          rpvp,		// vertical reverse periodic plan
          fpvc,		// vertical forward cosine plan
          rpvc,		// vertical reverse cosine plan
          fpvs,		// vertical forward sine plan
          rpvs;		// vertical reverse sine plan

static void physical2spectral_2d(fluid_real *in, fluid_complex *out);
static void spectral2physical_2d(fluid_complex *in, fluid_real *out);
static void physical2spectral_3d_p(fluid_real *in, fluid_complex *out);
static void physical2spectral_3d_c(fluid_real *in, fluid_complex *out);
static void physical2spectral_3d_s(fluid_real *in, fluid_complex *out);
static void spectral2physical_3d_p(fluid_complex *in, fluid_real *out);
static void spectral2physical_3d_c(fluid_complex *in, fluid_real *out);
static void spectral2physical_3d_s(fluid_complex *in, fluid_real *out);

int fourier_init() {
  int status = 0;

  rwork2d = fftw_alloc_real(grid_2d_nn_local*2);
  cwork2d = fftw_alloc_complex(grid_2d_nn_local);
  cworkv  = fftw_alloc_complex(grid_nz);
  rworkv  = fftw_alloc_real(grid_nz);

  fp2d =
    fftw_mpi_plan_dft_r2c_2d(grid_nx, grid_ny, rwork2d, cwork2d,
                             MPI_COMM_WORLD, FFTW_MEASURE);
  rp2d =
    fftw_mpi_plan_dft_c2r_2d(grid_nx, grid_ny, cwork2d, rwork2d,
                             MPI_COMM_WORLD, FFTW_MEASURE);

  if(grid_nd == 2) {
    return status;
  }

  assert(grid_nd == 3);

  fpvp = fftw_plan_dft_1d(grid_nz, cworkv, cworkv,
                          FFTW_FORWARD, FFTW_MEASURE);
  rpvp = fftw_plan_dft_1d(grid_nz, cworkv, cworkv,
                          FFTW_BACKWARD, FFTW_MEASURE);
  fpvs = fftw_plan_r2r_1d(grid_nz, rworkv, rworkv,
                          FFTW_RODFT10, FFTW_MEASURE);
  rpvs = fftw_plan_r2r_1d(grid_nz, rworkv, rworkv,
                          FFTW_RODFT01, FFTW_MEASURE);
  fpvc = fftw_plan_r2r_1d(grid_nz, rworkv, rworkv,
                          FFTW_REDFT10, FFTW_MEASURE);
  rpvc = fftw_plan_r2r_1d(grid_nz, rworkv, rworkv,
                          FFTW_REDFT01, FFTW_MEASURE);

  return status;
}

void fourier_finalize() {
  fftw_destroy_plan(fp2d);
  fftw_destroy_plan(rp2d);
  fftw_free(rwork2d);
  fftw_free(cwork2d);

  if(grid_nd == 3) {
    fftw_destroy_plan(fpvc);
    fftw_destroy_plan(rpvc);
    fftw_destroy_plan(fpvs);
    fftw_destroy_plan(rpvs);
    fftw_destroy_plan(fpvp);
    fftw_destroy_plan(rpvp);
    fftw_free(cworkv);
    fftw_free(rworkv);
  }
}

void physical2spectral_2d(fluid_real *in, fluid_complex *out) {
  fftw_mpi_execute_dft_r2c(fp2d,in,out);
}

void spectral2physical_2d(fluid_complex *in, fluid_real *out) {
  fftw_mpi_execute_dft_c2r(rp2d,in,out);
}

void physical2spectral_3d_p(fluid_real *in, fluid_complex *out) {
  int m, idx;

  fluid_real *in1;

  fluid_complex *out1;

  // do all horizontal transforms
  for(m = 0; m < grid_nz; m++) {
    in1 = &in[grid_2d_nn_local*2*m];
    out1 = &out[grid_2d_nn_local*m];
    fftw_mpi_execute_dft_r2c(fp2d,in1,out1);
  }

  // do vertical transforms
  for(idx = 0; idx < grid_2d_nn_local; idx++) {
    for(m = 0; m < grid_nz; m++) {
      cworkv[m] = out[grid_2d_nn_local*m + idx];
    }
    fftw_execute(fpvp);
    for(m = 0; m < grid_nz; m++) {
      out[grid_2d_nn_local*m + idx] = cworkv[m];
    }
  }
}

void physical2spectral_3d_c(fluid_real *in, fluid_complex *out) {
  int m, idx;

  fluid_real *in1;

  fluid_complex *out1;

  // do vertical transforms
  for(idx = 0; idx < grid_2d_nn_local*2; idx++) {
    for(m = 0; m < grid_nz; m++) {
      rworkv[m] = in[grid_2d_nn_local*2*m + idx];
    }
    fftw_execute(fpvc);
    for(m = 0; m < grid_nz; m++) {
      in[grid_2d_nn_local*2*m + idx] = rworkv[m];
    }
  }

  // do all horizontal transforms
  for(m = 0; m < grid_nz; m++) {
    in1 = &in[grid_2d_nn_local*2*m];
    out1 = &out[grid_2d_nn_local*m];
    fftw_mpi_execute_dft_r2c(fp2d,in1,out1);
  }
}

void physical2spectral_3d_s(fluid_real *in, fluid_complex *out) {
  int m, idx;

  fluid_real *in1;

  fluid_complex *out1;

  // do vertical transforms
  for(idx = 0; idx < grid_2d_nn_local*2; idx++) {
    for(m = 0; m < grid_nz; m++) {
      rworkv[m] = in[grid_2d_nn_local*2*m + idx];
    }
    fftw_execute(fpvs);
    for(m = 0; m < grid_nz; m++) {
      in[grid_2d_nn_local*2*m + idx] = rworkv[m];
    }
  }

  // do all horizontal transforms
  for(m = 0; m < grid_nz; m++) {
    in1 = &in[grid_2d_nn_local*2*m];
    out1 = &out[grid_2d_nn_local*m];
    fftw_mpi_execute_dft_r2c(fp2d,in1,out1);
  }
}

void spectral2physical_3d_p(fluid_complex *in, fluid_real *out) {
  int m, idx;

  fluid_complex *in1;

  fluid_real *out1;

  // do vertical transforms
  for(idx = 0; idx < grid_2d_nn_local; idx++) {
    for(m = 0; m < grid_nz; m++) {
      cworkv[m] = in[grid_2d_nn_local*m + idx];
    }
    fftw_execute(rpvp);
    for(m = 0; m < grid_nz; m++) {
      in[grid_2d_nn_local*m + idx] = cworkv[m];
    }
  }

  // do all horizontal transforms
  for(m = 0; m < grid_nz; m++) {
    in1 = &in[grid_2d_nn_local*m];
    out1 = &out[grid_2d_nn_local*2*m];
    fftw_mpi_execute_dft_c2r(rp2d,in1,out1);
  }
}

void spectral2physical_3d_c(fluid_complex *in, fluid_real *out) {
  int m, idx;

  fluid_complex *in1;

  fluid_real *out1;

  // do all horizontal transforms
  for(m = 0; m < grid_nz; m++) {
    in1 = &in[grid_2d_nn_local*m];
    out1 = &out[grid_2d_nn_local*2*m];
    fftw_mpi_execute_dft_c2r(rp2d,in1,out1);
  }

  // do vertical transforms
  for(idx = 0; idx < grid_2d_nn_local*2; idx++) {
    for(m = 0; m < grid_nz; m++) {
      rworkv[m] = out[grid_2d_nn_local*2*m + idx];
    }
    fftw_execute(rpvc);
    for(m = 0; m < grid_nz; m++) {
      out[grid_2d_nn_local*2*m + idx] = rworkv[m];
    }
  }

}

void spectral2physical_3d_s(fluid_complex *in, fluid_real *out) {
  int m, idx;

  fluid_complex *in1;

  fluid_real *out1;

  // do all horizontal transforms
  for(m = 0; m < grid_nz; m++) {
    in1 = &in[grid_2d_nn_local*m];
    out1 = &out[grid_2d_nn_local*2*m];
    fftw_mpi_execute_dft_c2r(rp2d,in1,out1);
  }

  // do vertical transforms
  for(idx = 0; idx < grid_2d_nn_local*2; idx++) {
    for(m = 0; m < grid_nz; m++) {
      rworkv[m] = out[grid_2d_nn_local*2*m + idx];
    }
    fftw_execute(rpvs);
    for(m = 0; m < grid_nz; m++) {
      out[grid_2d_nn_local*2*m + idx] = rworkv[m];
    }
  }

}

void physical2spectral(fluid_real *in, fluid_complex *out,
                       grid_vertical_layout_t layout) {
  fluid_real normalization;

  int idx;

  if(grid_nd == 2) {
    physical2spectral_2d(in,out);
    // normalize
    normalization = 1.0/(fluid_real)(grid_nx*grid_ny);
    for(idx = 0; idx < grid_2d_nn_local; idx++) {
      out[idx] *= normalization;
    }
    return;
  }

  switch(layout) {
    case GRID_VERTICAL_LAYOUT_PERIODIC:
      physical2spectral_3d_p(in,out);
      normalization = 1.0/(fluid_real)(grid_nx*grid_ny*grid_nz);
      break;

    case GRID_VERTICAL_LAYOUT_SINE:
      physical2spectral_3d_s(in,out);
      normalization = 1.0/(fluid_real)(grid_nx*grid_ny*grid_nz*2);
      break;

    case GRID_VERTICAL_LAYOUT_COSINE:
      physical2spectral_3d_c(in,out);
      normalization = 1.0/(fluid_real)(grid_nx*grid_ny*grid_nz*2);
      break;
 
    default:
      break;
  }

  // normalize
  for(idx = 0; idx < grid_3d_nn_local; idx++) {
    out[idx] *= normalization;
  }
}

void spectral2physical(fluid_complex *in, fluid_real *out,
                       grid_vertical_layout_t layout) {
  if(grid_nd == 2) {
    spectral2physical_2d(in,out);
    return;
  }

  switch(layout) {
    case GRID_VERTICAL_LAYOUT_PERIODIC:
      spectral2physical_3d_p(in,out);
      break;

    case GRID_VERTICAL_LAYOUT_SINE:
      spectral2physical_3d_s(in,out);
      break;

    case GRID_VERTICAL_LAYOUT_COSINE:
      spectral2physical_3d_c(in,out);
      break;
 
    default:
      break;
  }
}



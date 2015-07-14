#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>

#include "diag.h"
#include "grid.h"
#include "comm.h"
#include "state.h"

int diag_write(char *ofile_name) {
  FILE *ofile;

  ptrdiff_t idx,n;

  int ki, kj, kk, ksq, kbox;

  double kx, ky, kz, knorm, wgt;

  double ens_local  = 0.0,
         ens_global = 0.0,
         ke_local   = 0.0,
         ke_global  = 0.0,
         qabs;

  double *ke_profile;

  double *ke_profile_global;

  int ke_pnum = grid_nx/3;
  if(ke_pnum > grid_ny/3) ke_pnum = grid_ny/3;
  if(ke_pnum > grid_ny/3) ke_pnum = grid_ny/3;
  if(grid_nd == 3) {
    if(ke_pnum > grid_nz/3) ke_pnum = grid_nz/3;
  }

  ke_profile = calloc(ke_pnum,sizeof(double));

  if(my_task == master_task) {
    ke_profile_global = calloc(ke_pnum,sizeof(double));
  }

  for(idx = 0; idx < grid_nn_local; idx++) {
    ki = grid_ki[idx];
    kj = grid_kj[idx];
    kx = grid_kx[idx];
    ky = grid_ky[idx];
    if(grid_nd == 3) {
      kk = grid_kk[idx];
      kz = grid_kz[idx];
    }

    if(grid_nd == 2) {
      ksq = ki*ki + kj*kj;
      knorm = sqrt(kx*kx + ky*ky);
    }
    else {
      ksq = ki*ki + kj*kj + kk*kk;
      knorm = sqrt(kx*kx + ky*ky + kz*kz);
    }

    if(ksq < ke_pnum*ke_pnum && ksq > 0) {
      kbox = sqrt(ksq);
      qabs = cabs(kq[idx]);
      wgt  = grid_wgt[idx];
      ke_profile[kbox] += wgt*qabs*qabs/(knorm*knorm);
      ke_local += wgt*qabs*qabs/(knorm*knorm);
      ens_local += wgt*qabs*qabs;
    }
  }

  MPI_Reduce(&ke_local,  &ke_global,  1,
             MPI_DOUBLE, MPI_SUM, master_task, MPI_COMM_WORLD);
  MPI_Reduce(&ens_local, &ens_global, 1,
             MPI_DOUBLE, MPI_SUM, master_task, MPI_COMM_WORLD);

  for(n=0; n<ke_pnum; n++) {
    MPI_Reduce(&ke_profile[n], &ke_profile_global[n], 1,
               MPI_DOUBLE, MPI_SUM, master_task, MPI_COMM_WORLD);
  }

  if(my_task == master_task) {
    printf("KE/ENS: %1.15lf %1.15lf\n", ke_global, ens_global);
  }

  if(my_task == master_task) {
    ofile = fopen(ofile_name,"w");
    for(n = 0; n < ke_pnum; n++) {
      fprintf(ofile,"%i %1.15lf\n", n, ke_profile_global[n]);
    }

    fclose(ofile);
    free(ke_profile_global);
  }

  free(ke_profile);

  return 0;
}


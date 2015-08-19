#include <stdio.h>
#include <stdbool.h>
#include <iniparser.h>
#include <mpi.h>

#include "diag.h"
#include "comm.h"
#include "config.h"
#include "error.h"
#include "state.h"

void (*diag_write_function)();

int diag_write_set(void (*write_function)()) {
  diag_write_function = write_function;
  return 0;
}


int diag_write() {

  if(diag_write_function) {
    diag_write_function();
  }

  return 0;
}

write_energy(double *ke_out, double *pe_out) {
  int idx2d, idx3d, m;

  fluid_real u,v,w,b;
  fluid_real ke,ke_tot,ke_tot_g;
  fluid_real pe,pe_tot,pe_tot_g;

  pe_tot   = 0.0;
  ke_tot   = 0.0;
  ke_tot_g = 0.0;
  pe_tot_g = 0.0;
  for(idx2d = 0; idx2d < 2*grid_2d_nn_local; idx2d++) {
    for(m = 0; m < grid_nz; m++) {
      idx3d = idx2d + 2*grid_2d_nn_local*m;

      u = q[grid_3d_nn_local*2*0 + idx3d];
      v = q[grid_3d_nn_local*2*1 + idx3d];
      w = q[grid_3d_nn_local*2*2 + idx3d];
      b = q[grid_3d_nn_local*2*3 + idx3d];

      ke = 0.5*(u*u + v*v + w*w);
      pe = 0.5*(b*b);
      if(!grid_2d_buffer[idx2d]) {
        ke_tot += ke;
        pe_tot += pe;
      }
    }
  }

  MPI_Reduce(&ke_tot, &ke_tot_g,  1,
             MPI_DOUBLE, MPI_SUM, master_task, MPI_COMM_WORLD);
  MPI_Reduce(&pe_tot, &pe_tot_g,  1,
             MPI_DOUBLE, MPI_SUM, master_task, MPI_COMM_WORLD);

  *ke_out = ke_tot_g;
  *pe_out = pe_tot_g;

  return;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "comm.h"
#include "grid.h"
#include "state.h"
#include "time.h"
#include "diag.h"

int main(int argc, char *argv[]) {
  int i;

  char diag_file_name[256];

  MPI_Init(&argc, &argv);

  comm_init();
  grid_init();
  state_init();
  time_init();

  for(i  = 0; i < 10000; i++) {
    time_step();
    sprintf(diag_file_name,"ke_profile_%1.5lf.dat",time);
    diag_write(diag_file_name);
  }
  sprintf(diag_file_name,"ke_profile_%1.5lf.dat",time);
  diag_write(diag_file_name);

  state_write("outfile.bin");

  time_finalize();
  state_finalize();

  MPI_Finalize();

  return 0;
}


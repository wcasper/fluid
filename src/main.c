#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "time.h"
#include "diag.h"
#include "init.h"

int main(int argc, char *argv[]) {
  int i;

  char file_name[256];

  MPI_Init(&argc, &argv);

  init();

  for(i  = 0; i < 10000; i++) {
    if(i%10 == 0) {
      sprintf(file_name,"out/restart_%1.1lf.bin",time);
      state_write(file_name);
    }
    time_step();
    sprintf(file_name,"out/ke_profile_%1.1lf.dat",time);
    diag_write(file_name);
  }
  sprintf(file_name,"out/ke_profile_%1.1lf.dat",time);
  diag_write(file_name);
  sprintf(file_name,"out/restart_%1.1lf.bin",time);
  state_write(file_name);

  finalize();

  MPI_Finalize();

  return 0;
}


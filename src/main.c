#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#include "time.h"
#include "init.h"
#include "state.h"
#include "config.h"
#include "error.h"

int main(int argc, char *argv[]) {
  int i, status;

  char file_name[256];

  MPI_Init(&argc, &argv);

  status = init();
  error_check(&status, "error in init\n");
  if(status) return status;

  if(argc > 1) {
    config_file_name = argv[1];
    printf("Using config file %s\n", config_file_name);
  }

  for(i  = 0; i < 1000; i++) {
    if(i%1 == 0) {
      sprintf(file_name,"out/restart_%1.1lf.bin",time);
      state_write(file_name);
    }
    time_step();
  }

  sprintf(file_name,"out/restart_%1.1lf.bin",time);
  state_write(file_name);

  finalize();

  MPI_Finalize();

  return 0;
}


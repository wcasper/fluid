#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#include "fluid.h"
#include "time.h"
#include "init.h"
#include "state.h"
#include "config.h"
#include "error.h"

int main(int argc, char *argv[]) {
  int i, status;

  char file_name[__FLUID_STRLEN_MAX];

  MPI_Init(&argc, &argv);

  if(argc > 1) {
    config_file_name = argv[1];
    printf("Using config file %s\n", config_file_name);
  }

  status = init();
  error_check(&status, "error in init\n");
  if(status) return status;

  clock_t begin, end;
  double time_spent;

  begin = clock();

  for(i  = 0; i < 1000; i++) {
    if(i%1 == 0) {
      snprintf(file_name,__FLUID_STRLEN_MAX-1,"out/restart_%1.1lf.bin",time_model);
      state_write(file_name);
      snprintf(file_name,__FLUID_STRLEN_MAX-1,"out/vort_%1.1lf.bin",time_model);
      state_write_vort(file_name);
    }
    time_step();
  }

  end = clock();

  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Spent: %1.16lf\n", time_spent);


  snprintf(file_name,__FLUID_STRLEN_MAX-1,"out/restart_%1.1lf.bin",time_model);
  state_write(file_name);
  snprintf(file_name,__FLUID_STRLEN_MAX-1,"out/vort_%1.1lf.bin",time_model);
  state_write_vort(file_name);

  MPI_Barrier(MPI_COMM_WORLD);

  finalize();

  MPI_Finalize();

  return 0;
}


#include <stdio.h>
#include <stdlib.h>

#include "init.h"
#include "comm.h"
#include "grid.h"
#include "state.h"
#include "time.h"
#include "config.h"

int init() {
  int status = 0;

  // initialize MPI task numbers
  status = comm_init();
  if(status) {
    fprintf(stderr, "error in comm_init\n");
    return status;
  }

  // read in the configuration file
  status = config_read("example.ini");
  if(status) {
    fprintf(stderr, "error in config_read\n");
    return status;
  }

  // initialize grid
  status = grid_init();
  if(status) {
    fprintf(stderr, "error in grid_init\n");
    return status;
  }

  // initialize state
  status = state_init();
  if(status) {
    fprintf(stderr, "error in state_init\n");
    return status;
  }

  // initialize time
  status = time_init(TIME_STEP_MODEL_INS3D);
  if(status) {
    fprintf(stderr, "error in time_init\n");
    return status;
  }

  return status;
}

int finalize() {
  time_finalize();
  state_finalize();

  return 0;
}


#include "init.h"

#include "comm.h"
#include "grid.h"
#include "state.h"
#include "time.h"
#include "config.h"

int init() {
  // initialize MPI task numbers
  comm_init();

  // read in the configuration file
  config_read("config.ini");

  // initialize grid
  grid_init();

  // initialize state
  state_init();

  // initialize time
  time_init(TIME_STEP_MODEL_INS3D);

  return 0;
}

int finalize() {
  time_finalize();
  state_finalize();

  return 0;
}


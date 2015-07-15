#include "init.h"

#include "comm.h"
#include "grid.h"
#include "state.h"
#include "time.h"

int init() {
  // initialize MPI task numbers
  comm_init();

  // initialize grid
  grid_nx = 32;
  grid_ny = 32;
  grid_nz = 32;
  grid_init(GRID_LAYOUT_3D_PPP);

  // initialize state
  state_init(4);

  // initialize time
  time_init(TIME_STEP_MODEL_INS3D);

  return 0;
}

int finalize() {
  time_finalize();
  state_finalize();

  return 0;
}


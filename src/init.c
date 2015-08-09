#include <stdio.h>
#include <stdlib.h>

#include "init.h"
#include "comm.h"
#include "grid.h"
#include "fourier.h"
#include "state.h"
#include "model.h"
#include "error.h"
#include "time.h"

int init() {
  int status = 0;

  // initialize MPI task numbers
  status = comm_init();
  error_check(&status, "error in comm_init\n");
  if(status) return status;

  // initialize grid
  status = grid_init();
  error_check(&status, "error in grid_init\n");
  if(status) return status;

  // initialize fourier
  status = fourier_init();
  error_check(&status, "error in fourier_init\n");
  if(status) return status;

  // initialize state
  status = state_init();
  error_check(&status, "error in state_init\n");
  if(status) return status;

  // initialize model
  status = model_init();
  error_check(&status, "error in model_init\n");
  if(status) return status;

  // initialize time step
  status = time_init();
  error_check(&status, "error in time_init\n");
  if(status) return status;

  return status;
}

int finalize() {
  model_finalize();
  state_finalize();
  fourier_finalize();
  grid_finalize();

  return 0;
}


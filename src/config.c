#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iniparser.h>

#include "mpi.h"
#include "comm.h"
#include "config.h"
#include "grid.h"
#include "time.h"
#include "state.h"

int config_read(char *config_file_name) {
  dictionary *dict;

  char *file_name;

  int len;

  int status = 0;

  if(my_task == master_task) {
    // read the configuration file
    dict = iniparser_load(config_file_name);
    if(!dict) {
      status = 1;
    }
  }
  error_check(status, "error reading config file\n");
  if(status) return status;
  
  if(my_task == master_task) {
    // read in grid initialization data
    grid_nd = iniparser_getint(dict, "grid:nd", grid_nd);
    switch(grid_nd) {
      case(2):
        grid_nx = iniparser_getint(dict, "grid:nx", grid_nx);
        grid_ny = iniparser_getint(dict, "grid:ny", grid_ny);
        break;
      case(3):
        grid_nx = iniparser_getint(dict, "grid:nx", grid_nx);
        grid_ny = iniparser_getint(dict, "grid:ny", grid_ny);
        grid_nz = iniparser_getint(dict, "grid:nz", grid_nz);
        break;
      default:
        status = 1;
        break;
    }
  }
  error_check(status, "bad nd value in config file\n");
  if(status) return status;

  if(my_task == master_task) {
    grid_layout = iniparser_getint(dict, "grid:layout", grid_layout);
  
    // read in state initialization data
    nq = iniparser_getint(dict, "state:nq", nq);
    state_init_type = iniparser_getint(dict, "state:init", state_init_type);
    file_name = iniparser_getstring(dict, "state:rfile", NULL);
    len = 0;
    if(file_name) {
      len = strlen(file_name)+1;
      state_restart_file_name = calloc(len,sizeof(char));
    }
  
    // read in time initialization data
    time    = iniparser_getdouble(dict, "time:t0", 0.0);
    time_dt = iniparser_getdouble(dict, "time:dt", 1.0);
  
    // free the configuration file
    iniparser_freedict(dict);

  }

  // broadcast data from master_task to all other processors
  MPI_Bcast(&grid_nd,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_nx,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_ny,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_nz,1,MPI_INT,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&grid_layout,1,MPI_INT,master_task,MPI_COMM_WORLD);

  MPI_Bcast(&nq, 1, MPI_INT, master_task, MPI_COMM_WORLD);
  MPI_Bcast(&state_init_type, 1, MPI_INT, master_task, MPI_COMM_WORLD);
  MPI_Bcast(&len, 1, MPI_INT, master_task, MPI_COMM_WORLD);
  if(len) {
    state_restart_file_name = calloc(len,sizeof(char));
    MPI_Bcast(state_restart_file_name, len, MPI_CHAR,
              master_task, MPI_COMM_WORLD);
  }

  MPI_Bcast(&time, 1, MPI_DOUBLE, master_task, MPI_COMM_WORLD);
  MPI_Bcast(&time_dt, 1, MPI_DOUBLE, master_task, MPI_COMM_WORLD);

  return status;
}


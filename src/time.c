#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iniparser.h>
#include <mpi.h>

#include "time.h"
#include "fluid.h"
#include "comm.h"
#include "config.h"
#include "error.h"
#include "diag.h"
#include "state.h"

fluid_real time_model = 0.0;
fluid_real time_dt = 1e-0;
fluid_real time_step_dt = 1e-3;
fluid_real time_err_max = 1e-8;

fluid_real (*time_step_model)(fluid_real, fluid_real);

static int time_read_config();
static fluid_real time_getmax(fluid_complex *kfield);

int time_step_set(fluid_real (*step_function)(fluid_real, fluid_real)) {
  time_step_model = step_function;
  return 0;
}

int time_read_config() {
  int status = 0;
  dictionary *dict;

  if(my_task == master_task) {
    // read the configuration file
    dict = iniparser_load(config_file_name);
    if(!dict) {
      status = 1;
    }
  }
  error_check(&status, "error reading config file\n");
  if(status) return status;
  
  if(my_task == master_task) {
    // read in time initialization data
    time_model = iniparser_getdouble(dict, "time:t",  time_model);
    time_dt = iniparser_getdouble(dict, "time:dt", 1.0);
    time_step_dt = iniparser_getdouble(dict, "time:dt0", 1e-3);
    time_err_max = iniparser_getdouble(dict, "time:err_max", 1e-8);
    iniparser_freedict(dict);
  }

  MPI_Bcast(&time_model,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&time_dt,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&time_step_dt,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
  MPI_Bcast(&time_err_max,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);

  return status;
}


int time_init() {
  int status = 0;

  status = time_read_config();
  error_check(&status, "error in time_read_config\n");

  return status;
}

fluid_real time_getmax(fluid_complex *kfield) {
  int idx2d, idx3d, m;

  fluid_real u, umax, umax_global;

  umax = 0.0;
  for(idx2d = 0; idx2d < grid_2d_nn_local; idx2d++) {
    if(!grid_2d_dealias_mask[idx2d]) {
      for(m = 0; m < (grid_nz*2)/3; m++) {
        idx3d = idx2d + grid_2d_nn_local*m;

        u = cabs(kfield[idx3d]);

        umax = (u < umax) ? umax : u;
      }
    }
  }

  MPI_Reduce(&umax,  &umax_global,  1,
             MPI_DOUBLE, MPI_MAX, master_task, MPI_COMM_WORLD);
  return umax_global;
}


int time_step() {
  fluid_real err = 0.0,
         t1  = time_model + time_dt,
         dt, factor;

  bool is_boundary_step;

  fluid_real umax, vmax, wmax, bmax;

  while(time_model < t1) {
    if(time_model + time_step_dt > t1) {
      dt = t1-time_model;
      is_boundary_step = true;
    }
    else {
      dt = time_step_dt;
      is_boundary_step = false;
    }
    factor = 10.0;
    while(factor > 1.0) {
      err = time_step_model(dt,time_err_max);
      //printf("step err = %1.16lf, step size = %lf\n", err, dt);

      factor = err/time_err_max;
      if(factor > 1.0) {
        dt *= 0.9*pow(factor,-0.25);
        time_step_dt = dt;
      }

      umax = time_getmax(&kq[grid_3d_nn_local*0]);
      vmax = time_getmax(&kq[grid_3d_nn_local*1]);
      wmax = time_getmax(&kq[grid_3d_nn_local*2]);
      bmax = time_getmax(&kq[grid_3d_nn_local*3]);
      if(my_task == master_task) {
        printf("factor = %1.16lf, new dt = %lf, umax = %1.16lf, vmax = %1.16lf, wmax = %1.16lf, bmax = %1.16lf\n", factor, dt, umax, vmax, wmax, bmax);
      }
    }
    time_model += dt;
    if(factor > 1.89e-4 && !is_boundary_step) {
      time_step_dt *= 0.9*pow(factor,-0.2);
    }
    else if(!is_boundary_step) {
      time_step_dt *= 5.0;
    }

    if(time_step_dt > time_dt) time_step_dt = time_dt;
  }

  return 0;
}

int time_finalize() {

  return 0;
}


#include <stdio.h>
#include <stdbool.h>
#include <iniparser.h>
#include <mpi.h>

#include "time.h"
#include "comm.h"
#include "config.h"
#include "error.h"

double time = 0.0;
double time_dt = 1e-0;
double time_step_dt = 1e-3;
double time_err_max = 1e-8;

double (*time_step_model)(double, double);

static int time_read_config();

int time_step_set(double (*step_function)(double, double)) {
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
    // read in grid initialization data
    time = iniparser_getdouble(dict, "time:t",  time);
    time_dt = iniparser_getdouble(dict, "time:dt", 1.0);
    time_step_dt = iniparser_getdouble(dict, "time:dt0", 1e-3);
    time_err_max = iniparser_getdouble(dict, "time:err_max", 1e-8);
    iniparser_freedict(dict);
  }

  MPI_Bcast(&time,1,MPI_DOUBLE,master_task,MPI_COMM_WORLD);
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

int time_step() {
  double err = 0.0,
         t1  = time + time_dt,
         dt;

  bool is_boundary_step;

  while(time < t1) {
    if(time + time_step_dt > t1) {
      dt = t1-time;
      is_boundary_step = true;
    }
    else {
      dt = time_step_dt;
      is_boundary_step = false;
    }
    err = time_err_max + 1.0;
    while(err >= time_err_max) {
      err = time_step_model(dt,time_err_max);
      printf("step err = %1.16lf, step size = %lf\n", err, dt);

      if(err >= time_err_max) {
        dt *= 0.5;
        time_step_dt = dt;
      }
    }
    time += dt;
    if(err < 0.1*time_err_max && !is_boundary_step) {
      time_step_dt *= 2.0;
    }
    if(time_step_dt > time_dt) time_step_dt = time_dt;
  }

  return 0;
}

int time_finalize() {

  return 0;
}


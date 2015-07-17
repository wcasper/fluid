#include "time.h"
#include "stdio.h"

double time = 0.0;
double time_dt = 1e-0;
double time_step_dt = 1e-3;
double time_err_max = 1e-8;

double (*time_step_model)(double, double);

int time_step_set(double (*step_function)(double, double)) {
  time_step_model = step_function;
  return 0;
}

int time_step() {
  double err = 0.0,
         t1  = time + time_dt,
         dt;

  while(time < t1) {
    if(time + time_step_dt > t1) {
      dt = t1-time;
    }
    else {
      dt = time_step_dt;
    }
    err = time_err_max + 1.0;
    while(err >= time_err_max) {
      err = time_step_model(dt,time_err_max);

      if(err >= time_err_max) {
        dt *= 0.5;
        time_step_dt = dt;
      }
    }
    time += dt;
    if(err < 0.1*time_err_max && dt == time_step_dt) {
      time_step_dt *= 2.0;
    }
    if(time_step_dt > time_dt) time_step_dt = time_dt;
  }

  return 0;
}

int time_finalize() {

  return 0;
}


#include "time.h"
#include "ins2d.h"
#include "ins3d.h"
#include "stdio.h"

double time = 0.0;
double time_dt = 1e-1;
double time_step_dt = 1e-3;
double time_err_max = 1e-6;

double (*time_step_model)(double, double);

int time_init(int time_step_model_type) {
  switch(time_step_model_type) {
    case(TIME_STEP_MODEL_INS2D):
      ins2d_init();
      time_step_model = &ins2d_step_rk4_adaptive;
      break;

    case(TIME_STEP_MODEL_INS3D):
      ins3d_init();
      time_step_model = &ins3d_step_rk4_adaptive;
      break;
  }

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
      printf("%1.16lf %1.16lf %1.16lf\n", err/time_err_max, time_step_dt, dt);
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
  ins2d_finalize();

  return 0;
}


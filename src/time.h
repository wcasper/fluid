#ifndef __TIME_H
#define __TIME_H

#include "fluid.h"

extern fluid_real time_model;
extern fluid_real time_dt;

int time_step_set(fluid_real (*step_function)(fluid_real, fluid_real));

int time_init();
int time_finalize();

int time_step();

#endif


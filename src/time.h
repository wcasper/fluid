#ifndef __TIME_H
#define __TIME_H

extern double time;
extern double time_dt;

int time_step_set(double (*step_function)(double, double));

int time_init();
int time_finalize();

int time_step();

#endif


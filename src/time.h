#ifndef __TIME_H
#define __TIME_H

extern double time;
extern double time_dt;

int time_init(int time_step_model_type);
int time_finalize();

int time_step();

#define TIME_STEP_MODEL_INS2D 2
#define TIME_STEP_MODEL_INS3D 3

#endif


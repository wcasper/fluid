#ifndef __INS2D_H
#define __INS2D_H

extern int ins2d_step_type;

int ins2d_init();
int ins2d_finalize();

double ins2d_step_rk4_adaptive(double dt, double max_err_bnd);
#endif


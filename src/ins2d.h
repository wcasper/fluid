#ifndef __INCOMPRESSIBLE_NS_2D_H
#define __INCOMPRESSIBLE_NS_2D_H

extern int ins2d_step_type;

int ins2d_init();
double ins2d_step_rk4_adaptive(double dt, double max_err_bnd);
int ins2d_finalize();

#endif


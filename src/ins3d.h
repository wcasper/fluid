#ifndef __INCOMPRESSIBLE_NS_3D_H
#define __INCOMPRESSIBLE_NS_3D_H

extern int ins3d_step_type;

int ins3d_init();
double ins3d_step_rk4_adaptive(double dt, double max_err_bnd);
int ins3d_finalize();

#endif


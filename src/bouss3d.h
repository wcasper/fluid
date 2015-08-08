#ifndef __BOUSS_3D_H
#define __BOUSS_3D_H

extern int bouss3d_step_type;

int bouss3d_init();
int bouss3d_finalize();

double bouss3d_step_rk4_adaptive(double dt, double max_err_bnd);
#endif


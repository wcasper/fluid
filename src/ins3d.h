#ifndef __INS3D_3D_H
#define __INS3D_3D_H

extern int ins3d_step_type;

int ins3d_init();
int ins3d_finalize();

double ins3d_step_rk4_adaptive(double dt, double max_err_bnd);

#endif


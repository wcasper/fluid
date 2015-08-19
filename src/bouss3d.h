#ifndef __BOUSS_3D_H
#define __BOUSS_3D_H

#include "fluid.h"

extern int bouss3d_step_type;

int bouss3d_init();
int bouss3d_finalize();

fluid_real bouss3d_step_rk4_adaptive(fluid_real dt, fluid_real max_err_bnd);

#endif


#ifndef __INS3D_3D_H
#define __INS3D_3D_H

#include "fluid.h"

extern int ins3d_step_type;

int ins3d_init();
int ins3d_finalize();

fluid_real ins3d_step_rk4_adaptive(fluid_real dt, fluid_real max_err_bnd);

#endif


#ifndef __INS2D_H
#define __INS2D_H

#include "fluid.h"

extern int ins2d_step_type;

int ins2d_init();
int ins2d_finalize();

fluid_real ins2d_step_rk4_adaptive(fluid_real dt, fluid_real max_err_bnd);
#endif


#ifndef __FOURIER_H
#define __FOURIER_H

#include "fluid.h"
#include "grid.h"

int fourier_init();
void fourier_finalize();

void physical2spectral(fluid_real *in, fluid_complex *out,
                       grid_vertical_layout_t layout);
void spectral2physical(fluid_complex *in, fluid_real *out,
                       grid_vertical_layout_t layout);

#endif


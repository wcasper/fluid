#ifndef __FOURIER_H
#define __FOURIER_H

#include "complex.h"
#include "grid.h"

int fourier_init();
void fourier_finalize();

void physical2spectral(double *in, double complex *out,
                       grid_vertical_layout_t layout);
void spectral2physical(double complex *in, double *out,
                       grid_vertical_layout_t layout);

#endif


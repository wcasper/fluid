#ifndef __STATE_H
#define __STATE_H

#include <math.h>
#include <complex.h>

extern double *q;		// state variables in physical space
extern double complex *kq;	// state variables in spectral space

extern ptrdiff_t nq;		// number of state variables

int state_init();
int state_read(char *ifile_name);
int state_write(char *ifile_name);
int state_physical2spectral();
int state_spectral2physical();
int state_finalize();

#endif


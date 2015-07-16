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

typedef enum {
  STATE_INIT_TYPE_RESTART,
  STATE_INIT_TYPE_PATCHES_2D,
  STATE_INIT_TYPE_PATCHES_3D,
} state_init_type_t;

extern state_init_type_t state_init_type;
extern char * state_restart_file_name;

#endif


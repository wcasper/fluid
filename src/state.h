#ifndef __STATE_H
#define __STATE_H

#include <math.h>
#include <complex.h>

#include "grid.h"

extern double *q;		// state variables in physical space
extern double complex *kq;	// state variables in spectral space

extern ptrdiff_t nq;		// number of state variables

extern grid_vertical_layout_t * state_layout; // vertical expansion types

int state_init();
int state_read(char *ifile_name);
int state_write(char *ofile_name);
int state_physical2spectral();
int state_spectral2physical();
int state_finalize();

typedef enum {
  STATE_INIT_TYPE_RESTART,
  STATE_INIT_TYPE_PATCHES_2D,
  STATE_INIT_TYPE_PATCHES_3D,
  STATE_INIT_TYPE_BOUSS3D_TEST1,
  STATE_INIT_TYPE_INS3D_TEST1
} state_init_type_t;

extern state_init_type_t state_init_type;
extern char * state_restart_file_name;

int state_write_vort(char *ofile_name);

#endif


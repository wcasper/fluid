#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#include "spline2d.h"

// calculate a degree 3 spline of 2 dim periodic data:
// in each cell, data is approximated by degree 3 polynomial
// f(x,y) = p00*N0(x)N0(y) + p01*N0(x)N1(y) + p10*N1(x)N0(y) + ...
// where Ni(x) = binom(3,i)*x^i(1-x)^(3-i)
spline2d_t spline2d_init(double *data, int m, int n) {
  int i, j, l1, l2, idx;

  double  *d;

  double complex *kcx, *kcy, *kd, *kp;

  double complex wx[4], wy[4];

  double complex kcxx, kcyy;

  fftw_plan cxplan, cyplan, dplan, iplan;

  spline2d_t myspline;

  myspline.nx = m;
  myspline.ny = n;

  myspline.cells = calloc(m*n,sizeof(spline2d_cell_t));

  d  = calloc(m*(n/2+1)*2,sizeof(double));

  kcx = calloc(m,sizeof(double complex));
  kcy = calloc(n,sizeof(double complex));
  kd  = calloc(m*(n/2+1),sizeof(double complex));
  kp  = calloc(m*(n/2+1),sizeof(double complex));

  cxplan = fftw_plan_dft_1d(m, kcx, kcx, FFTW_BACKWARD, FFTW_MEASURE);
  cyplan = fftw_plan_dft_1d(n, kcy, kcy, FFTW_BACKWARD, FFTW_MEASURE);

  dplan = fftw_plan_dft_r2c_2d(m, n, d, kd, FFTW_MEASURE);
  iplan = fftw_plan_dft_c2r_2d(m, n, kp, d, FFTW_MEASURE);

  // initialize kcx,kcy
  kcx[0] = 1.0 + 0.0*I;
  kcx[1] = 4.0 + 0.0*I;
  kcx[2] = 1.0 + 0.0*I;
  kcy[0] = 1.0 + 0.0*I;
  kcy[1] = 4.0 + 0.0*I;
  kcy[2] = 1.0 + 0.0*I;
  fftw_execute(cxplan);
  fftw_execute(cyplan);

  // calculate kd
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      idx = i*(n/2 + 0)*2 + j;
      d[idx] = data[i*n + j];
    }
  }
  fftw_execute(dplan); // kd now holds fft of data

  // calculate p's
  for(l1 = 0; l1 < 4; l1++) {
    for(l2 = 0; l2 < 4; l2++) {
      for(i = 0; i < m; i++) {
        for(j = 0; j < n/2 + 1; j++) {
          idx = i*(n/2 + 1) + j;

          // weights calcualted here!!
          wx[0] = 1.0;
          wx[1] = cexp(2.0*M_PI*I*i/(double)m)*4.0;
          wx[1]+= cexp(4.0*M_PI*I*i/(double)m)*2.0;
          wx[2] = cexp(2.0*M_PI*I*i/(double)m)*2.0;
          wx[2]+= cexp(4.0*M_PI*I*i/(double)m)*4.0;
          wx[3] = cexp(2.0*M_PI*I*i/(double)m);
          wy[0] = 1.0;
          wy[1] = cexp(2.0*M_PI*I*j/(double)n)*4.0;
          wy[1]+= cexp(4.0*M_PI*I*j/(double)n)*2.0;
          wy[2] = cexp(2.0*M_PI*I*j/(double)n)*2.0;
          wy[2]+= cexp(4.0*M_PI*I*j/(double)n)*4.0;
          wy[3] = cexp(2.0*M_PI*I*j/(double)n);

          if(l1 == 0 || l1 == 3) {
            kcxx = 1.0;
          }
          else {
            kcxx = kcx[i];
          }
          if(l2 == 0 || l2 == 3) {
            kcyy = 1.0;
          }
          else {
            kcyy = kcy[j];
          }

          if(cabs(kcxx*kcyy) > 0) {
            kp[idx] = kd[idx]*wx[l1]*wy[l2]/(kcxx*kcyy);
          }
        }
      }
      fftw_execute(iplan); // d holds p[i*n + j][l1*4 + l2] unnormalized
      for(i = 0; i < m; i++) {
        for(j = 0; j < n; j++) {
          myspline.cells[i*n + j].p[l1][l2]
            = d[i*(n/2+0)*2 + j]/(double)(m*n);
        }
      }
    }
  }

  fftw_destroy_plan(cxplan);
  fftw_destroy_plan(cyplan);
  fftw_destroy_plan(dplan);
  fftw_destroy_plan(iplan);

  free(kcx);
  free(kcy);
  free(kd);
  free(kp);

  return myspline;
}

void spline2d_free(spline2d_t *myspline) {
  if(myspline->cells) free(myspline->cells);

  myspline->nx = 0;
  myspline->ny = 0;
}

double spline2d_eval(spline2d_t *myspline, int i, int j, double t1, double t2) {
  int status = 0;

  double val = 0.0;

  int l1, l2;

  double p1[4], p2[4];

  spline2d_cell_t *cell;

  // sanity check
  if(i < 0 || j < 0 || i > myspline->nx || j > myspline->ny) {
    fprintf(stderr, "array index out of bounds\n");
    status = 1;
  }
  if(myspline->cells == NULL) {
    fprintf(stderr, "spline2d_eval called on uninitialized spline\n");
    status = 1;
  }
  if(status) return -999999.0;

  cell = &(myspline->cells[i*myspline->ny + j]);

  p1[0] = (1.0-t1)*(1.0-t1)*(1.0-t1);
  p1[1] = t1*(1.0-t1)*(1.0-t1)*3.0;
  p1[2] = t1*t1*(1.0-t1)*3.0;
  p1[3] = t1*t1*t1;
  p2[0] = (1.0-t2)*(1.0-t2)*(1.0-t2);
  p2[1] = t2*(1.0-t2)*(1.0-t2)*3.0;
  p2[2] = t2*t2*(1.0-t2)*3.0;
  p2[3] = t2*t2*t2;

  for(l1 = 0; l1 < 4; l1++) {
    for(l2 = 0; l2 < 4; l2++) {
      val += cell->p[l1][l2]*p1[l1]*p2[l2];
    }
  }

  return val;
}


double spline2d_eval_xderiv(spline2d_t *myspline, int i, int j, double t1, double t2) {
  int status = 0;

  double val = 0.0;

  int l1, l2;

  double dp1[4], p2[4];

  spline2d_cell_t *cell;

  // sanity check
  if(i < 0 || j < 0 || i > myspline->nx || j > myspline->ny) {
    fprintf(stderr, "array index out of bounds\n");
    status = 1;
  }
  if(myspline->cells == NULL) {
    fprintf(stderr, "spline2d_eval called on uninitialized spline\n");
    status = 1;
  }
  if(status) return -999999.0;

  cell = &(myspline->cells[i*myspline->ny + j]);

  dp1[0] = -3.0*(1.0-t1)*(1.0-t1);
  dp1[1] = 3.0*(1.0-t1)*(1.0-t1) - 6.0*t1*(1.0-t1);
  dp1[2] = 6.0*t1*(1.0-t1) - 3.0*t1*t1;
  dp1[3] = 3.0*t1*t1;
  p2[0] = (1.0-t2)*(1.0-t2)*(1.0-t2);
  p2[1] = t2*(1.0-t2)*(1.0-t2)*3.0;
  p2[2] = t2*t2*(1.0-t2)*3.0;
  p2[3] = t2*t2*t2;

  for(l1 = 0; l1 < 4; l1++) {
    for(l2 = 0; l2 < 4; l2++) {
      val += cell->p[l1][l2]*dp1[l1]*p2[l2];
    }
  }

  return val;
}

double spline2d_eval_yderiv(spline2d_t *myspline, int i, int j, double t1, double t2) {
  int status = 0;

  double val = 0.0;

  int l1, l2;

  double p1[4], dp2[4];

  spline2d_cell_t *cell;

  // sanity check
  if(i < 0 || j < 0 || i > myspline->nx || j > myspline->ny) {
    fprintf(stderr, "array index out of bounds\n");
    status = 1;
  }
  if(myspline->cells == NULL) {
    fprintf(stderr, "spline2d_eval called on uninitialized spline\n");
    status = 1;
  }
  if(status) return -999999.0;

  cell = &(myspline->cells[i*myspline->ny + j]);

  p1[0] = (1.0-t1)*(1.0-t1)*(1.0-t1);
  p1[1] = t1*(1.0-t1)*(1.0-t1)*3.0;
  p1[2] = t1*t1*(1.0-t1)*3.0;
  p1[3] = t1*t1*t1;
  dp2[0] = -3.0*(1.0-t2)*(1.0-t2);
  dp2[1] = 3.0*(1.0-t2)*(1.0-t2) - 6.0*t2*(1.0-t2);
  dp2[2] = 6.0*t2*(1.0-t2) - 3.0*t2*t2;
  dp2[3] = 3.0*t2*t2;

  for(l1 = 0; l1 < 4; l1++) {
    for(l2 = 0; l2 < 4; l2++) {
      val += cell->p[l1][l2]*p1[l1]*dp2[l2];
    }
  }

  return val;
}



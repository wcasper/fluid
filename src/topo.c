#include <stdlib.h>
#include <math.h>

#include "topo.h"
#include "comm.h"
#include "grid.h"
#include "spline2d.h"

double *topo_hb;	// height of bottom topography
double *topo_d2b; 	// signed distance to bottom
double *topo_refl_x;    // x position of virtual pt of exterior pt
double *topo_refl_y;    // y position of virtual pt of exterior pt
double *topo_refl_z;    // z position of virtual pt of exterior pt
int    *topo_refl_task; // task w/ data of virt. pt of exterior pt

spline2d_t topo_spline;

static void closest_pt_on_bottom(int i, int j, int k,
                                 double *x, double *y,
                                 double *z, double *d);

int topo_init() {
  int i,j,k, idx;

  double x, y, z, d;

  topo_hb        = calloc(grid_2d_nn, sizeof(double));
  topo_d2b       = calloc(grid_3d_nn, sizeof(double));
  topo_refl_x    = calloc(grid_3d_nn, sizeof(double));
  topo_refl_y    = calloc(grid_3d_nn, sizeof(double));
  topo_refl_z    = calloc(grid_3d_nn, sizeof(double));
  topo_refl_task = calloc(grid_3d_nn, sizeof(int));

  // initialize the bottom as rolling hills homogeneous in y
  for(i = 0; i < grid_nx; i++) {
    for(j = 0; j < grid_ny; j++) {
      topo_hb[i*grid_ny + j] = 0.10 + 0.05*cos(4.0*M_PI*i/(double)grid_nx);
      //topo_hb[i*grid_ny + j] = 0.1;
      topo_hb[i*grid_ny + j] *= grid_lz;
    }
  }

  topo_spline = spline2d_init(topo_hb, grid_nx, grid_ny);

  // calulate distance to bottom and reflection points
  for(i = 0; i < grid_nx; i++) {
    for(j = 0; j < grid_ny; j++) {
      for(k = 0; k < grid_nz; k++) {
        idx = grid_nz*(grid_ny*i + j) + k;
        closest_pt_on_bottom(i,j,k,&x,&y,&z,&d);
        topo_d2b[idx] = d;
        if(d < 0.0) {
          // calculate reflected point
          topo_refl_x[idx] = 2.0*x-grid_lx*i/(double)grid_nx;
          topo_refl_y[idx] = 2.0*y-grid_ly*j/(double)grid_ny;
          topo_refl_z[idx] = 2.0*z-grid_lz*k/(double)grid_ny;

          // impose periodicity
          if(topo_refl_x[idx] < 0.0) topo_refl_x[idx] += grid_lx;
          if(topo_refl_y[idx] < 0.0) topo_refl_y[idx] += grid_ly;
          if(topo_refl_z[idx] < 0.0) topo_refl_z[idx] += grid_lz;
          if(topo_refl_x[idx] > grid_lx) topo_refl_x[idx] -= grid_lx;
          if(topo_refl_y[idx] > grid_ly) topo_refl_y[idx] -= grid_ly;
          if(topo_refl_z[idx] > grid_lz) topo_refl_z[idx] -= grid_lz;

          // determine processor reflected point is on
        }
      }
    }
  }

  return 0;
}

int topo_finalize() {
  spline2d_free(&topo_spline);

  free(topo_hb);
  free(topo_d2b);
  free(topo_refl_x);
  free(topo_refl_y);
  free(topo_refl_z);
  free(topo_refl_task);

  return 0;
}

void closest_pt_on_bottom(int i, int j, int k,
                          double *x, double *y,
                          double *z, double *d) {

  const int max_iter = 10;

  int ci, cj, iter;

  double rx, ry;

  double x0, y0, z0,
         x1, y1, z1;

  double dx, dy;

  double dbdx, dbdy, prod, dsq0, dsq1;

  x0 = i*grid_dx;
  y0 = j*grid_dy;
  z0 = k*grid_dz;

  x1 = x0;
  y1 = y0;
  z1 = topo_hb[i*grid_ny + j];

  ci = i;
  cj = j;
  rx = 0.0;
  ry = 0.0;
  dsq1 = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0);
  for(iter = 0; iter < max_iter; iter++) {

    dbdx = spline2d_eval_xderiv(&topo_spline, ci, cj, rx, ry);
    dbdy = spline2d_eval_yderiv(&topo_spline, ci, cj, rx, ry);

    dbdx *= grid_dx;
    dbdy *= grid_dy;

    prod = dbdx*(x1-x0) + dbdy*(y1-y0) - (z1-z0);
    prod /= dbdx*dbdx + dbdy*dbdy + 1.0;

    // calculate step to new x,y guess
    dx = -((x1-x0)- dbdx*prod)*0.5;
    dy = -((y1-y0)- dbdy*prod)*0.5;

    // prevent large jumps
    if(dx >  0.5*grid_dx) dx =  0.5*grid_dx;
    if(dx < -0.5*grid_dx) dx = -0.5*grid_dx;
    if(dy >  0.5*grid_dy) dy =  0.5*grid_dy;
    if(dy < -0.5*grid_dy) dy = -0.5*grid_dy;

    x1 += dx;
    y1 += dy;
    rx += dx/grid_dx;
    ry += dy/grid_dy;
    if(rx < 0.0) {
      rx = 1.0+rx;
      ci = (ci-1+grid_nx)%grid_nx;
    }
    else if(rx > 1.0) {
      rx = rx - 1.0;
      ci = (ci+1)%grid_nx;
    }
    if(ry < 0.0) {
      ry = 1.0+ry;
      cj = (cj-1+grid_ny)%grid_ny;
    }
    else if(ry > 1.0) {
      ry = ry - 1.0;
      cj = (cj+1)%grid_ny;
    }
    z1 = spline2d_eval(&topo_spline, ci, cj, rx, ry);
    dsq0 = dsq1;
    dsq1 = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0);
    if(dsq1 > dsq0+1e-14) {
      break;
    }
    else if(dsq0-dsq1 < 1e-14){
      break;;
    }
  }

  *x = x1;
  *y = y1;
  *z = z1;
  *d = sqrt(dsq1);
  if(topo_hb[i*grid_ny + j]-z0 > 0.0) {
    *d *= -1.0;
  }
}






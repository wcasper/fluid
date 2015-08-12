#!/usr/bin/python

import sys
import numpy
import restart
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

nx = 32;
ny = 32;
nz = 32;

# read in density
t = int(sys.argv[1]);
u = restart.restart_read("/scratch/wcasper/out/restart_%i.0.bin" % t, 1, [nz,ny,nx]);
#u = restart.restart_read("/scratch/wcasper/out08112015/restart_%i.0.bin" % t, 1, [nz,ny,nx]);
#u = restart.restart_read("../src/advection.bin", 3, [nz,ny,nx]);
#u = restart.restart_read("../src/advection.bin", 0, [nz,ny,nx]);

umean = numpy.sum(u[:,0,:],axis=1)/nx;

min = numpy.min(u);
max = numpy.max(u);

print(min,max)

#print(umean)
#print(sum(umean))

x = numpy.arange(0,1,1.0/nx);
y = numpy.arange(0,1,1.0/ny);
z = numpy.arange(0.5/nz,1,1.0/nz);

## mask land
#for i in range(nx):
#  for j in range(ny):
#    for k in range(nz):
#      foo = 0.1  + 0.05*numpy.cos(4.0*numpy.pi*x[i]);
#      if(z[k] < foo):
#        u[k,j,i] = numpy.nan;
land = numpy.cos(4.0*numpy.pi*x);
land = 0.1 + 0.05*land + 0.5/nz;

cmap = plt.cm.get_cmap("winter")
cmap.set_under("magenta")
cmap.set_over("yellow")
levels = numpy.arange(min,max,(max-min)*.001);
plt.contourf(x,z[:],u[:,0,:], levels, extend='both')
#plt.contourf(x,y,u[6,:,:], levels, extend='both')
plt.colorbar();
#plt.plot(umean,z, "ro");
#plt.plot(x,land, "r-");
plt.xlim(0,1);
plt.ylim(0,1);
plt.show();


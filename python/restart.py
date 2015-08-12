# python module to read a restart file

import numpy
import struct

def restart_read(filename, rec, dims = []):
  nd = len(dims);
  if (nd == 2):
    recl = dims[0]*int(dims[1]/2 + 1)*2;
  elif (nd == 3):
    recl = dims[0]*dims[1]*int(dims[2]/2 + 1)*2;
  else:
    print("Error in read_restart: cannot handle dimension %i" %nd);
    return;

  print(recl)

  recl_bytes = recl*8;
  file = open(filename, "rb");
  file.seek(recl_bytes*rec);
  blob = file.read(recl_bytes);
  file.close();

  data = struct.unpack("d"*recl,blob);
  
  out  = numpy.empty(dims, dtype=float);

  if(nd == 3):
    nx = dims[0];
    ny = dims[1];
    nz = dims[2];

    for i in range(0, dims[0]):
      for j in range(0, dims[1]):
        for k in range(0, dims[2]):
          out[i,j,k] = data[2*int(ny/2 + 1)*(nx*k + i) + j];

  if(nd == 2):
    nx = dims[0];
    ny = dims[1];

    for i in range(0, dims[0]):
      for j in range(0, dims[1]):
          out[i,j] = data[2*int(ny/2 +1)*i + j];

  return out;






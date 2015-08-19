#ifndef __DIAG_H
#define __DIAG_H

int diag_write_set(void (*write_function)());

int diag_write();

void write_energy(double *ke, double *pe);

#endif


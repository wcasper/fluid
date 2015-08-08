#ifndef __COMM_H
#define __COMM_H

#include <stdlib.h>
#include <complex.h>

#include "grid.h"

extern int my_task;
extern int master_task;
extern int num_tasks;

int comm_init();

int scatter_global_array(void *local, void *global, size_t size);
int gather_global_array(void *local, void *global, size_t size);

#endif


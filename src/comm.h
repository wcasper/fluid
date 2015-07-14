#ifndef __COMM_H
#define __COMM_H

#include <complex.h>

extern int my_task;
extern int master_task;
extern int num_tasks;

int comm_init();

int scatter_global_array(void *local, void *global, size_t size, int grid_type);
int gather_global_array(void *local, void *global, size_t size, int grid_type);

#endif


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mpi.h"
#include "comm.h"
#include "grid.h"
#include "error.h"

int my_task;
int num_tasks;
int master_task;


int comm_init() {
  master_task = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_task); 

  return 0;
}

int scatter_global_array(void *local, void *global, size_t size) {
  int stat = 0;

  int task, tag1, tag2, tag3,
      task_n0, //starting index of task
      task_nn; //number of indices in the current task

  MPI_Status status;

  ptrdiff_t idx, num;

  tag1  = 1;
  tag2  = 2;
  tag3  = 3;

  if(my_task == master_task) {
    num = grid_3d_nn_local*2*size;
    memcpy(local, global, num);
  }

  if(my_task == master_task) {
    for(task = 1; task < num_tasks; task++) {
      MPI_Recv(&task_n0,1,MPI_INT,task,
		tag1,MPI_COMM_WORLD,&status);
      MPI_Recv(&task_nn,1,MPI_INT,task,
		tag2,MPI_COMM_WORLD,&status);

      idx = (grid_ny/2 + 1)*2*task_n0*size;
      if(grid_nd == 3){
        idx *= grid_nz;
      }
      num = task_nn*2*size;

      MPI_Send(&((char *)global)[idx], num, MPI_CHAR,
		task, tag3, MPI_COMM_WORLD);
    }
  }
  else {
    MPI_Send(&grid_2d_n0_local,1,MPI_INT,master_task,
	     tag1,MPI_COMM_WORLD);
    MPI_Send(&grid_3d_nn_local,1,MPI_INT,master_task,
             tag2,MPI_COMM_WORLD);

    num = grid_3d_nn_local*2*size;

    MPI_Recv(local, num, MPI_CHAR, master_task,
	     tag3, MPI_COMM_WORLD, &status);
  }


  return stat;
}

int gather_global_array(void *local, void *global, size_t size) {
  int stat = 0;

  int task, tag1, tag2, tag3,
      task_n0, //starting index of task
      task_nn; //number of indices in the current task

  int m;

  MPI_Status status;

  ptrdiff_t idx, num;

  tag1  = 1;
  tag2  = 2;
  tag3  = 3;

  if(my_task == master_task) {
    num = grid_2d_nn_local*2*size;
    if(grid_nd == 2) {
      idx = (grid_ny/2 + 1)*2*grid_2d_n0_local*size;
      memcpy(&global[idx], local, num);
    }
    else {
      for(m = 0; m < grid_nz; m++) {
        idx = (grid_ny/2 + 1)*2*grid_2d_n0_local*size;
        idx +=(grid_ny/2 + 1)*2*grid_nx*m*size;
        memcpy(&((char *)global)[idx], local, num);
      }
    }
  }

  if(my_task == master_task) {
    for(task = 1; task < num_tasks; task++) {
      MPI_Recv(&task_n0,1,MPI_INT,task,
		tag1,MPI_COMM_WORLD,&status);
      MPI_Recv(&task_nn,1,MPI_INT,task,
		tag2,MPI_COMM_WORLD,&status);

      num = task_nn*2*size;
      if(grid_nd == 2) {
        idx = (grid_ny/2 + 1)*2*task_n0*size;
        MPI_Recv(&((char *)global)[idx], num, MPI_CHAR, task,
                 tag3, MPI_COMM_WORLD, &status);
      }
      else {
        for(m = 0; m < grid_nz; m++) {
          idx = (grid_ny/2 + 1)*2*task_n0*size;
          idx +=(grid_ny/2 + 1)*2*grid_nx*m*size;
          MPI_Recv(&((char *)global)[idx], num, MPI_CHAR, task,
		   tag3+m, MPI_COMM_WORLD, &status);
        }
      }
    }
  }

  else {
    task_n0 = grid_2d_n0_local;
    task_nn = grid_2d_nn_local;
    MPI_Send(&task_n0,1,MPI_INT,master_task,
             tag1,MPI_COMM_WORLD);
    MPI_Send(&task_nn,1,MPI_INT,master_task,
             tag2,MPI_COMM_WORLD);

    num = grid_2d_nn_local*2*size;
    if(grid_nd == 2) {
      MPI_Send(local, num, MPI_CHAR, master_task,
               tag3, MPI_COMM_WORLD);
    }
    else {
      for(m = 0; m < grid_nz; m++) {
        idx = grid_2d_nn_local*2*m*size;
        MPI_Send(&((char *)local)[idx], num, MPI_CHAR,
                 master_task, tag3+m, MPI_COMM_WORLD);
      }
    }
  }

  return stat;
}



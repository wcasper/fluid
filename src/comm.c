#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mpi.h"
#include "comm.h"
#include "grid.h"

int my_task;
int num_tasks;
int master_task;


int comm_init() {
  master_task = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_task); 

  return 0;
}

int scatter_global_array(void *local, void *global, size_t size, grid_type_t grid_type) {
  int task, tag1, tag2, tag3,
      task_n0, //starting index of task
      task_nn; //number of indices in the current task

  MPI_Status status;

  ptrdiff_t idx, num;

  tag1  = 1;
  tag2  = 2;
  tag3  = 3;

  if(my_task == master_task) {
    switch(grid_type) {
      case GRID_TYPE_PHYSICAL:          
        num = grid_nn_local*2*size;
        memcpy(local, global, num);
        break;

      case GRID_TYPE_SPECTRAL:
        num = grid_nn_local*size;
        memcpy(local, global, num);
        break;

      default:
        fprintf(stderr,"Unknown grid type\n");
        exit(EXIT_FAILURE);
    }

    for(task = 1; task < num_tasks; task++) {
      MPI_Recv(&task_n0,1,MPI_INT,task,
		tag1,MPI_COMM_WORLD,&status);
      MPI_Recv(&task_nn,1,MPI_INT,task,
		tag2,MPI_COMM_WORLD,&status);

      if(grid_nd == 2) {
        idx = (grid_ny/2 + 1)*size;
      }
      else {
        idx = grid_ny*(grid_nz/2 + 1)*size;
      }
      switch(grid_type) {
        case GRID_TYPE_PHYSICAL:          
          idx*= task_n0*2;
          num = task_nn*2*size;
          break;

        case GRID_TYPE_SPECTRAL:
          idx*= task_n0;
          num = task_nn*size;
          break;

        default:
          fprintf(stderr,"Unknown grid type\n");
          exit(EXIT_FAILURE);
      }

      MPI_Send(&((char *)global)[idx], num, MPI_CHAR,
		task, tag3, MPI_COMM_WORLD);
    }
  }

  else {
      MPI_Send(&grid_n0_local,1,MPI_INT,master_task,
		tag1,MPI_COMM_WORLD);
      MPI_Send(&grid_nn_local,1,MPI_INT,master_task,
		tag2,MPI_COMM_WORLD);

      switch(grid_type) {
        case GRID_TYPE_PHYSICAL:          
          num = grid_nn_local*2*size;
          break;

        case GRID_TYPE_SPECTRAL:          
          num = grid_nn_local*size;
          break;

        default:
          fprintf(stderr,"Unknown grid type\n");
          exit(EXIT_FAILURE);
      }

      MPI_Recv(local, num, MPI_CHAR, master_task,
		tag3, MPI_COMM_WORLD, &status);
  }


  return 0;
}

int gather_global_array(void *local, void *global, size_t size, grid_type_t grid_type) {
  int task, tag1, tag2, tag3,
      task_n0, //starting index of task
      task_nn; //number of indices in the current task

  MPI_Status status;

  ptrdiff_t idx, num;

  tag1  = 1;
  tag2  = 2;
  tag3  = 3;

  if(my_task == master_task) {
    switch(grid_type) {
      case GRID_TYPE_PHYSICAL:          
        num = grid_nn_local*2*size;
        memcpy(global, local, num);
        break;

      case GRID_TYPE_SPECTRAL:
        num = grid_nn_local*size;
        memcpy(global, local, num);
        break;

      default:
        fprintf(stderr,"Unknown grid type\n");
        exit(EXIT_FAILURE);
    }

    for(task = 1; task < num_tasks; task++) {
      MPI_Recv(&task_n0,1,MPI_INT,task,
		tag1,MPI_COMM_WORLD,&status);
      MPI_Recv(&task_nn,1,MPI_INT,task,
		tag2,MPI_COMM_WORLD,&status);

      if(grid_nd == 2) {
        idx = (grid_ny/2 + 1)*size;
      }
      else {
        idx = grid_ny*(grid_nz/2 + 1)*size;
      }
      switch(grid_type) {
        case GRID_TYPE_PHYSICAL:          
          idx*= task_n0*2;
          num = task_nn*2*size;
          break;

        case GRID_TYPE_SPECTRAL:
          idx*= task_n0;
          num = task_nn*size;
          break;

        default:
          fprintf(stderr,"Unknown grid type\n");
          exit(EXIT_FAILURE);
      }

      MPI_Recv(&((char *)global)[idx], num, MPI_CHAR, task,
		tag3, MPI_COMM_WORLD, &status);
    }
  }

  else {
      MPI_Send(&grid_n0_local,1,MPI_INT,master_task,
		tag1,MPI_COMM_WORLD);
      MPI_Send(&grid_nn_local,1,MPI_INT,master_task,
		tag2,MPI_COMM_WORLD);

      switch(grid_type) {
        case GRID_TYPE_PHYSICAL:          
          num = grid_nn_local*2*size;
          break;

        case GRID_TYPE_SPECTRAL:          
          num = grid_nn_local*size;
          break;

        default:
          fprintf(stderr,"Unknown grid type\n");
          exit(EXIT_FAILURE);
      }

      MPI_Send(local, num, MPI_CHAR,
		master_task, tag3, MPI_COMM_WORLD);
  }


  return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>

#include "error.h"
#include "comm.h"

int error_check(int *status, char *error_message) {
  char ems[ERROR_MESSAGE_MAX_LENGTH+1];

  int ems_len;

  // if *status == 0 for master task, do not print error message
  MPI_Bcast(status, 1, MPI_INT, master_task, MPI_COMM_WORLD);
  if(*status == 0) {
    return 0;
  }

  // otherwise, master_task prints error message and all tasks exit
  if(my_task == master_task) {
    if(error_message) {
      ems_len = strlen(error_message);
      if(ems_len > ERROR_MESSAGE_MAX_LENGTH) {
         ems_len = ERROR_MESSAGE_MAX_LENGTH;
      }

      strncpy(ems, error_message, ems_len);
      fprintf(stderr, "ERROR: %s\n", error_message);
    }
  }

  return 0;
}

int error_exit(int *status, char *error_message) {
  char ems[ERROR_MESSAGE_MAX_LENGTH+1];

  int ems_len;

  // if *status == 0 for master task, do not exit program
  MPI_Bcast(status, 1, MPI_INT, master_task, MPI_COMM_WORLD);
  if(*status == 0) {
    return 0;
  }

  // otherwise, master_task prints error message and all tasks exit
  if(my_task == master_task) {
    if(error_message) {
      ems_len = strlen(error_message);
      if(ems_len > ERROR_MESSAGE_MAX_LENGTH) {
         ems_len = ERROR_MESSAGE_MAX_LENGTH;
      }

      strncpy(ems, error_message, ems_len);
      fprintf(stderr, "ERROR: %s (crashing)\n", error_message);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  exit(*status);

  return 0;
}


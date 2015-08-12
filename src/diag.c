#include <stdio.h>
#include <stdbool.h>
#include <iniparser.h>
#include <mpi.h>

#include "diag.h"
#include "comm.h"
#include "config.h"
#include "error.h"

void (*diag_write_function)();

int diag_write_set(void (*write_function)()) {
  diag_write_function = write_function;
  return 0;
}


int diag_write() {

  if(diag_write_function) {
    diag_write_function();
  }

  return 0;
}



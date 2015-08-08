#include <iniparser.h>
#include <mpi.h>

#include "model.h"
#include "comm.h"
#include "config.h"
#include "time.h"
#include "ins2d.h"
#include "bouss3d.h"
#include "error.h"

model_type_t model_type = MODEL_INS2D;

static int model_read_config();

int model_init() {
  int status = 0;

  status = model_read_config();
  error_check(&status, "error in model_read_config\n");
  if(status) return status;

  switch(model_type) {
    case MODEL_INS2D:
      time_step_set(ins2d_step_rk4_adaptive);
      status = ins2d_init();
      error_check(&status,"error in ins2d_init\n");
      break;
    case MODEL_BOUSS3D:
      time_step_set(bouss3d_step_rk4_adaptive);
      status = bouss3d_init();
      error_check(&status,"error in bouss3d_init\n");
      break;
    default:
      status = 1;
      error_check(&status, "unrecognized model type\n");
  } 

  return status;
}

int model_finalize() {

  switch(model_type) {
    case MODEL_INS3D:
      bouss3d_finalize();
      break;
    default:
      break;
  } 

  return 0;
}

int model_read_config() {
  int status = 0, type;
  dictionary *dict;

  if(my_task == master_task) {
    // read the configuration file
    dict = iniparser_load(config_file_name);
    if(!dict) {
      status = 1;
    }
  }
  error_check(&status, "error reading config file\n");
  if(status) return status;
  
  if(my_task == master_task) {
    // read in model initialization data
    type = iniparser_getint(dict, "model:type", model_type);
  }

  iniparser_freedict(dict);

  MPI_Bcast(&type,1,MPI_INT,master_task,MPI_COMM_WORLD);

  model_type = (model_type_t)type;

  return status;
}


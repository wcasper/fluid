#include "model.h"
#include "time.h"
#include "ins2d.h"
#include "ins3d.h"
#include "error.h"

model_type_t model_type = -1;

int model_init() {
  int status = 0;

  switch(model_type) {
    case MODEL_INS2D:
      time_step_set(ins2d_step_rk4_adaptive);
      status = ins2d_init();
      error_check(&status,"error in ins2d_init\n");
      break;
    case MODEL_INS3D:
      time_step_set(ins3d_step_rk4_adaptive);
      status = ins3d_init();
      error_check(&status,"error in ins3d_init\n");
      break;
    default:
      status = 1;
      error_check(&status, "unrecognized model type\n");
  } 

  return status;
}

int model_finalize() {

  switch(model_type) {
    case MODEL_INS2D:
      ins2d_finalize();
      break;
    case MODEL_INS3D:
      ins3d_finalize();
      break;
    default:
      break;
  } 

  return 0;
}

#ifndef __MODEL_H
#define __MODEL_H

int model_init();
int model_finalize();

typedef enum {
  MODEL_INS2D,
  MODEL_INS3D,
} model_type_t;

extern model_type_t model_type;

#endif


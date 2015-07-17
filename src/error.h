#ifndef __ERROR_H
#define __ERROR_H

#define ERROR_MESSAGE_MAX_LENGTH 255

int error_check(int *status, char *error_message);

int error_exit(int *status, char *error_message);

#endif


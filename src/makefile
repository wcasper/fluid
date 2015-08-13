TARGET = fluid
LIBS = -lm -L/usr/projects/climate/wcasper/lib -lfftw3_mpi -lfftw3 -liniparser
INCLUDES = -I/usr/projects/climate/wcasper/include
CC = mpicc
CFLAGS = -g -Wall -Wextra -w3
#CFLAGS = -g -Wall -Wextra

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.c, %.o, $(wildcard *.c))
HEADERS = $(wildcard *.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)

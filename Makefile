# Makefile for the GPUDirect RDMA benchmark program
PROJECT	 := rdma-bench
DIS_HOME := /opt/DIS
CU_HOME	 := /usr/local/cuda


# Locate project files
CC_SRC	 := $(wildcard src/*.cc)
CU_SRC	 := $(wildcard src/*.cu)
HEADERS	 := $(wildcard src/*.h)
OBJECTS	 := $(CC_SRC:%.cc=%.o) $(CU_SRC:%.cu=%.o)


# Compiler and linker settings
CC	 := /usr/bin/g++
NVCC 	 := $(CU_HOME)/bin/nvcc
CFLAGS	 := -Wall -Wextra -D_REENTRANT -g -O0
INCLUDE	 := -I$(DIS_HOME)/include -I$(DIS_HOME)/include/dis -I$(CU_HOME)/include
LDLIBS	 := -lsisci -lpthread -lcuda

ifneq ($(shell getconf LONG_BIT),)
	LDFLAGS := -L$(DIS_HOME)/lib64 -L$(CU_HOME)/lib64
else
	LDFLAGS := -L$(DIS_HOME)/lib -L$(CU_HOME)/lib
endif


# Compilation targets
.PHONY: all clean

all: $(PROJECT)


clean:
	-$(RM) $(PROJECT) $(OBJECTS)


$(PROJECT): $(OBJECTS)
	$(NVCC) -ccbin $(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)


# How to compile CUDA
%.o: %.cu $(HEADERS)
	$(NVCC) -std=c++11 -x cu -ccbin $(CC) -Xcompiler "$(CFLAGS)" $(INCLUDE) -o $@ $< -c


# How to compile C++
%.o: %.cc $(HEADERS)
	$(CC) -x c++ -std=c++11 $(CFLAGS) $(INCLUDE) -o $@ $< -c


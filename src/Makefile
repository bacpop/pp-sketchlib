
CXXFLAGS=-Wall -Wextra -std=c++14
ifdef DEBUG
  CXXFLAGS+= -O0 -g
else ifdef PROFILE
  CXXFLAGS+= -O2 -g -Wl,--compress-debug-sections=none
else
  CXXFLAGS+= -O2
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXXFLAGS+= -DMKL_ILP64 -m64
	LIBLOC=/nfs/users/nfs_j/jl11/pathogen_nfs/large_installations/miniconda3/envs/conda_py36
	LALIBS=-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -lm -ldl
endif
ifeq ($(UNAME_S),Darwin)
	LIBLOC=/Users/jlees/miniconda3/envs/pp_sketch
	LALIBS=-llapack -lblas -pthread
endif

CPPFLAGS=-I../dlib -I../HighFive/include -I../rollinghashcpp -I$(LIBLOC)/include -I$(LIBLOC)/include/eigen3 -D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1
LDFLAGS=-L$(LIBLOC)/lib -lz -lbz2 -lhdf5_cpp -lhdf5 -lxxhash $(LALIBS)

PROGRAMS=sketch_test read_test

SKETCH_OBJS=dist.o reference.o seqio.o sketch.o database.o countmin.o api.o linear_regression.o link_function.o

all: $(PROGRAMS)

static: $(STATIC_PROGRAMS)

clean:
	$(RM) *.o ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

sketch_test: $(SKETCH_OBJS) main.o
	$(LINK.cpp) $^ -o $@

read_test: $(SKETCH_OBJS) read_test.o
	$(LINK.cpp) $^ -o $@

.PHONY: all clean install

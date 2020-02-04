
CXXFLAGS=-Wall -Wextra -std=c++14 -march=native
ifdef DEBUG
  CXXFLAGS+= -O0 -g
else ifdef PROFILE
  CXXFLAGS+= -O2 -g
else
  CXXFLAGS+= -O2
endif

UNAME_S := $(shell uname -s)
LIBLOC = ${CONDA_PREFIX}
ifeq ($(UNAME_S),Linux)
	CXXFLAGS+= -DMKL_ILP64 -m64
	ifdef PROFILE
		CXXFLAGS+= -Wl,--compress-debug-sections=none
	endif
	LALIBS=-lmkl_rt -lpthread -lm -ldl
	# LALIBS=-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -lm -ldl
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		LDFLAGS=-Wl,-as-needed
	endif
endif
ifeq ($(UNAME_S),Darwin)
	LALIBS=-llapack -lopenblas -pthread
endif

CPPFLAGS=-I../dlib -I../HighFive/include -I$(LIBLOC)/include -I$(LIBLOC)/include/eigen3 -D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1
LDFLAGS+= -L$(LIBLOC)/lib -lz -lbz2 -lhdf5_cpp -lhdf5 $(LALIBS)

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


CXXFLAGS=-Wall -Wextra -O2 -std=c++1z -DMKL_ILP64 -m64
ifdef DEBUG
  CXXFLAGS=-Wall -Wextra -O0 -g -std=c++1z
endif


CPPFLAGS=-I../dlib -I../HighFive/include -I/Users/jlees/miniconda3/envs/pp_sketch/include -I/Users/jlees/miniconda3/envs/pp_sketch/include/eigen3 -D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1

LDFLAGS=-L/Users/jlees/miniconda3/envs/pp_sketch/lib -lz -lbz2 -lhdf5_cpp -lhdf5 -lxxhash -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -lm -ldl

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

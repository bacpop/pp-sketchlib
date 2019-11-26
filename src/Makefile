
CXXFLAGS=-Wall -Wextra -O2 -std=c++14
ifdef DEBUG
  CXXFLAGS=-Wall -Wextra -O0 -g -std=c++14
endif


CPPFLAGS=-I../dlib -I../rollinghashcpp -I../HighFive/include -I/Users/jlees/miniconda3/envs/py37env/include -D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1

LDFLAGS=-L/Users/jlees/miniconda3/envs/py37env/lib -lz -lbz2 -lhdf5 -pthread

PROGRAMS=sketch_test

SKETCH_OBJS=dist.o reference.o seqio.o sketch.o database.o

all: $(PROGRAMS)

static: $(STATIC_PROGRAMS)

clean:
	$(RM) *.o ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

sketch_test: $(SKETCH_OBJS) main.o
	$(LINK.cpp) $^ -o $@



.PHONY: all clean install

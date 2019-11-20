
CXXFLAGS=-Wall -O2 -std=c++14

CPPFLAGS=-I../seqan3/include -I../dlib -I../rollinghashcpp -D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1

PROGRAMS=sketch_test

SKETCH_OBJS=dist.o reference.o seqio.o sketch.o

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

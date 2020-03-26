
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
	LDFLAGS=-Wl,-as-needed
endif
ifeq ($(UNAME_S),Darwin)
	LALIBS=-llapack -lopenblas -pthread
endif

CPPFLAGS=-I../HighFive/include -I$(LIBLOC)/include -I$(LIBLOC)/include/eigen3
LDFLAGS+= -L$(LIBLOC)/lib -lz -lbz2 -lhdf5_cpp -lhdf5 $(LALIBS)

PROGRAMS=sketch_test read_test

SKETCH_OBJS=dist.o reference.o seqio.o sketch.o database.o countmin.o api.o linear_regression.o
CUDA_OBJS=dist.cu

# web specific options
web: CXX = emcc
web: CXXFLAGS += -s USE_ZLIB=1
web: LDFLAGS = -s USE_ZLIB=1 --preload-file 12673_8_24.fa

all: $(PROGRAMS)

web: web_sketch.o seqio.o sketch.o countmin.o
	$(LINK.cpp) $^ -o pp-sketch.html

clean:
	$(RM) *.o ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

sketch_test: $(SKETCH_OBJS) $(CUDA_OBJS) main.o
	$(LINK.cpp) $^ -o $@

read_test: $(SKETCH_OBJS) read_test.o
	$(LINK.cpp) $^ -o $@

.PHONY: all clean install

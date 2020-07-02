
CXXFLAGS+=-Wall -Wextra -std=c++14 -fopenmp -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
ifdef DEBUG
  CXXFLAGS+= -O0 -g
else ifdef PROFILE
  CXXFLAGS+= -O2 -g -flto -fno-fat-lto-objects -fPIC -fvisibility=hidden
else
  CXXFLAGS+= -O3 -flto -fno-fat-lto-objects -fPIC -fvisibility=hidden
endif

UNAME_S := $(shell uname -s)
LIBLOC = ${CONDA_PREFIX}
LDLIBS = -lz -lhdf5_cpp -lhdf5 -lopenblas -llapack -lgomp
ifeq ($(UNAME_S),Linux)
	CXXFLAGS+= -m64
	ifdef PROFILE
		CXXFLAGS+= -Wl,--compress-debug-sections=none
	endif
	LDLIBS+= -lpthread -lgfortran -lm -ldl -lrt
	LDFLAGS=-Wl,-as-needed
endif
ifeq ($(UNAME_S),Darwin)
	LDLIBS+= -pthread
endif

CPPFLAGS+=-I$(LIBLOC)/include -I$(LIBLOC)/include/eigen3
LDFLAGS+= -L$(LIBLOC)/lib
CUDA_LDLIBS=-lcudadevrt -lcudart_static $(LDLIBS)

CUDA_LDFLAGS =-L$(LIBLOC)/lib -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
CUDAFLAGS =-Xcompiler -fPIC -Xptxas -dlcm=ca --cudart static --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_75,code=sm_75

PYTHON_LIB = pp_sketchlib$(shell python3-config --extension-suffix)

# python specific options
python: CPPFLAGS += -DGPU_AVAILABLE -DPYTHON_EXT -DNDEBUG -Dpp_sketchlib_EXPORTS -DVERSION_INFO=\"1.5.0\" $(shell python3 -m pybind11 --includes)

PROGRAMS=sketch_test matrix_test read_test

SKETCH_OBJS=dist.o matrix_ops.o reference.o seqio.o sketch.o database.o countmin.o api.o linear_regression.o rng.o random_match.o
GPU_SKETCH_OBJS=gpu_api.o
CUDA_OBJS=dist.cu.o

all: $(PROGRAMS)

clean:
	$(RM) *.o *.so version.h ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

sketch_test: $(SKETCH_OBJS) main.o
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

matrix_test: $(SKETCH_OBJS) matrix_test.o
	$(LINK.cpp) $^ -o $@ $(LDLIBS)

read_test: $(SKETCH_OBJS) read_test.o
	$(LINK.cpp) $^ -o $@ $(LDLIBS)

version.h:
	cat sketch.cpp sketch.hpp ntHashIterator.hpp nthash.hpp stHashIterator.hpp | openssl sha1 | awk '{print "#define SKETCH_VERSION \"" $$2 "\""}' > version.h

database.o: version.h

python: $(PYTHON_LIB)

$(PYTHON_LIB): $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) sketchlib_bindings.o
	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) -shared $^ device_link.o -o $(PYTHON_LIB) $(CUDA_LDLIBS)

install_python: python
	install -d $(PYTHON_LIB_PATH)
	install $(PYTHON_LIB) $(PYTHON_LIB_PATH)

dist.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c dist.cu -o $@

.PHONY: all clean install python install_python

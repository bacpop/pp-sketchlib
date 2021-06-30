CXXFLAGS+=-Wall -Wextra -std=c++14 -fopenmp -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -fPIC
ifdef DEBUG
  CXXFLAGS+= -O0 -g
  CUDAFLAGS = -g -G
else ifdef PROFILE
  CXXFLAGS+= -O2 -g -flto -fno-fat-lto-objects -fvisibility=hidden
  CUDAFLAGS = -O2 -pg -lineinfo
else
  CXXFLAGS+= -O3 -flto -fno-fat-lto-objects -fvisibility=hidden
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

CPPFLAGS+=-I"." -I$(LIBLOC)/include -I$(LIBLOC)/include/eigen3
LDFLAGS+= -L$(LIBLOC)/lib
CUDA_LDLIBS=-lcudadevrt -lcudart_static $(LDLIBS)

CUDA_LDFLAGS =-L$(LIBLOC)/lib -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
CUDAFLAGS +=-Xcompiler -fPIC -Xptxas -dlcm=ca --cudart static --relocatable-device-code=true -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
ifdef GPU
	CXXFLAGS += -DGPU_AVAILABLE
	CUDAFLAGS += -gencode arch=compute_86,code=sm_86
	CUDA_LDFLAGS += -L/usr/local/cuda-11.1/lib64
endif

PYTHON_LIB = pp_sketchlib$(shell python3-config --extension-suffix)

# python specific options
python: CPPFLAGS += -DGPU_AVAILABLE -DPYTHON_EXT -DNDEBUG -Dpp_sketchlib_EXPORTS $(shell python3 -m pybind11 --includes)

PROGRAMS=sketch_test matrix_test read_test gpu_dist_test

SKETCH_OBJS=dist/dist.o dist/matrix_ops.o reference.o sketch/seqio.o sketch/sketch.o database/database.o sketch/countmin.o api.o dist/linear_regression.o random/rng.o random/random_match.o
GPU_SKETCH_OBJS=gpu/gpu_api.o
CUDA_OBJS=gpu/dist.cu.o gpu/sketch.cu.o gpu/device_reads.cu.o gpu/gpu_countmin.cu.o gpu/device_memory.cu.o

# web specific options
web: CXX = emcc
# optimised compile options
# NB turn exceptions back on for testing
web: CXXFLAGS = -O3 -s ASSERTIONS=1  \
				--closure 1 \
				-DNOEXCEPT \
				-DJSON_NOEXCEPTION \
				-s DISABLE_EXCEPTION_CATCHING=1 \
				-fno-exceptions \
				-flto --bind -s STRICT=1 \
				-s ALLOW_MEMORY_GROWTH=1 \
				-s USE_ZLIB=1 \
				-s MODULARIZE=1 \
				-s "EXPORTED_FUNCTIONS=['_malloc']" \
				-s 'EXTRA_EXPORTED_RUNTIME_METHODS=["FS"]' \
				-s EXPORT_NAME=WebSketch \
				-Wall -Wextra -std=c++14
web: CPPFLAGS += -DWEB_SKETCH
web: LDFLAGS = -lnodefs.js -lworkerfs.js

WEB_OUT=web/web_sketch
WEB_OBJS=${WEB_OUT}.js ${WEB_OUT}.html ${WEB_OUT}.wasm

web: web/web_sketch.o sketch/seqio.o sketch/sketch.o sketch/countmin.o
	$(LINK.cpp) $^ -o ${WEB_OUT}.js
	sed -i.old '1s;^;\/* eslint-disable *\/;' ${WEB_OUT}.js

all: $(PROGRAMS)

clean:
	$(RM) $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) $(WEB_OBJS) *.o *.so version.h ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

sketch_test: $(SKETCH_OBJS) test/main.o
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

matrix_test: $(SKETCH_OBJS) test/matrix_test.o
	$(LINK.cpp) $^ -o $@ $(LDLIBS)

read_test: $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) test/read_test.o
	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ device_link.o -o $@ $(CUDA_LDLIBS)

gpu_dist_test: $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) test/gpu_dist_test.o
	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ device_link.o -o $@ $(CUDA_LDLIBS)

version.h:
	cat sketch/*.cpp sketch/*.hpp gpu/sketch.cu | openssl sha1 | awk '{print "#define SKETCH_VERSION \"" $$2 "\""}' > version.h

database/database.o: version.h

web/web_sketch.o: version.h

python: $(PYTHON_LIB)

$(PYTHON_LIB): $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) sketchlib_bindings.o
	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) -shared $^ device_link.o -o $(PYTHON_LIB) $(CUDA_LDLIBS)

install_python: python
	install -d $(PYTHON_LIB_PATH)
	install $(PYTHON_LIB) $(PYTHON_LIB_PATH)

gpu/dist.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/dist.cu -o $@

gpu/sketch.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/sketch.cu -o $@

gpu/device_memory.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/device_memory.cu -o $@

gpu/device_reads.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/device_reads.cu -o $@

gpu/gpu_countmin.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/gpu_countmin.cu -o $@

.PHONY: all clean install python install_python web

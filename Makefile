SHELL := /bin/bash

IDIR = include
SDIR = src
TDIR = test_src
BINDIR = bin
ODIR = obj
OUTDIR = output
LIBDIR = lib

FFTLIB = fft

SUBDIRS = $(IDIR) $(SDIR) $(TDIR) $(BINDIR) $(OUTDIR) $(ODIR) $(ODIR)/test $(LIBDIR)

CXX := g++

O_LEVEL = 3

CVWARNINGS_SUPPRESS = -Wno-deprecated-anon-enum-enum-conversion -Wno-deprecated-enum-enum-conversion

CV_FLAGS = `pkg-config --cflags opencv4` $(CVWARNINGS_SUPPRESS)
CV_LIBS = `pkg-config --libs opencv4`

# IPYTHON := /opt/homebrew/opt/python@3.10/Frameworks/Python.framework/Versions/3.10/include/python3.10
# LPYTHON := /opt/homebrew/opt/python@3.10/Frameworks/Python.framework/Versions/3.10/lib
# INUMPY := /opt/homebrew/lib/python3.10/site-packages/numpy/core/include

# find the python include and lib directories
IPYTHON := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
LPYTHON := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
INUMPY  := $(shell python3 -c "import numpy; print(numpy.get_include())")
PYTHONINTERP := $(shell python3 -c "import sys; print(sys.executable)")

# get the last part of the python lib directory
LIBPYTHON := $(shell basename $(LPYTHON))

# remove last part of LPYTHON
LPYTHON := $(shell dirname $(LPYTHON))

# if macos add these flags
ifeq ($(shell uname), Darwin)
	CXXOMPFLAGS := -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
	LOMPFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
else
	CXXOMPFLAGS := -fopenmp
	LOMPFLAGS := -fopenmp
endif

NVCC := $(shell command -v nvcc 2> /dev/null)
NVCC_FLAGS :=  -I$(IDIR) -std=c++20 -arch=sm_80 -O$(O_LEVEL) -Xcompiler -Wall,-Wextra
NVCC_LIBS := -lcuda

# if nvcc is installed, use it
ifneq ($(NVCC),)
	CUDA_ROOT_DIR = /opt/cuda
	CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
	# CUDA include directory:
	CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
	# CUDA linking libraries:
	CUDA_LINK_LIBS= -lcudart
endif

CXXFLAGS= -I$(IDIR) -std=c++20 -g -O$(O_LEVEL) -Wall -Wextra $(CXXOMPFLAGS) $(CV_FLAGS)
LDFLAGS = $(LOMPFLAGS) $(CV_LIBS) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CUFILES = $(wildcard $(SDIR)/*.cu)
CUFILES = $(notdir $(_CUFILES))

_CXXFILES = $(wildcard $(SDIR)/*.cpp)
CXXFILES = $(notdir $(_CXXFILES))

# if the user specified to not use cuda blank out the nvcc variable
ifeq ($(USE_CUDA), 0)
	NVCC :=
endif

# if nvcc is not installed
ifeq ($(NVCC),)
	_OBJ = $(_CXXFILES:.cpp=.o)
	USE_CUDA = 0
else
	_OBJ = $(_CXXFILES:.cpp=.o) $(_CUFILES:.cu=.o)
	USE_CUDA = 1
endif

OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

_TESTCXXFILES = $(wildcard $(TDIR)/*.cpp)
TESTCXXFILES = $(notdir $(_TESTCXXFILES))

_TESTOBJ = $(_TESTCXXFILES:.cpp=.o)
TEST_OBJ = $(patsubst $(TDIR)/%,$(ODIR)/test/%,$(_TESTOBJ))

TEST_TARGET := $(BINDIR)/test
LIBRARY_TARGET := $(LIBDIR)/libfft.a

$(LIBRARY_TARGET): $(OBJ) | subdirs
	@echo "Building library"
	ar rcs $@ $^

build_test: subdirs $(TEST_TARGET)

test: build_test $(TEST_TARGET)
	@mkdir -p $(TDIR)/data
	@./$(TEST_TARGET) $(TDIR)/data
	@$(PYTHONINTERP) $(TDIR)/plot_scaling_tests_results.py  $(TDIR)/data
# @$(PYTHONINTERP) $(TDIR)/generate_fft.py $(TDIR)/data/signal.txt         $(TDIR)/data/numpy_fft.txt
# @$(PYTHONINTERP) $(TDIR)/compare_fft.py  $(TDIR)/data/comparison_fft.txt $(TDIR)/data/numpy_fft.txt

$(OBJ) : | subdirs

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS) -DUSE_CUDA=$(USE_CUDA)

$(TEST_TARGET): $(TEST_OBJ) | $(LIBRARY_TARGET)
	$(CXX) -o $@ $^ $(LDFLAGS) -L$(LIBDIR) -l$(FFTLIB)

$(ODIR)/test/%.o: $(TDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR)/%.o : $(SDIR)/%.cu $(IDIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

clean:
	rm -f $(ODIR)/*.o $(TARGET) $(TEST_TARGET) $(LIBRARY_TARGET) $(ODIR)/test/*.o 

.PHONY: clean run subdirs

subdirs: | $(SUBDIRS)

$(SUBDIRS):
	mkdir -p $@
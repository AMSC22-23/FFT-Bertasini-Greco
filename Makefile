SHELL := /bin/bash

IDIR = include
SDIR = src
TDIR = test_src
BINDIR = bin
ODIR = obj
OUTDIR = output
LIBDIR = lib

SUBDIRS = $(IDIR) $(SDIR) $(TDIR) $(BINDIR) $(OUTDIR) $(ODIR) $(ODIR)/test $(LIBDIR)

CXX := g++

O_LEVEL = 3

CVWARNINGS_SUPPRESS = -Wno-deprecated-anon-enum-enum-conversion

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

CXXFLAGS= -I$(IDIR) -I$(IPYTHON) -I$(INUMPY) -std=c++20 -g -O$(O_LEVEL) -Wall -Wextra $(CXXOMPFLAGS) $(CV_FLAGS)
LDFLAGS = -L$(LPYTHON) -l$(LIBPYTHON) $(LOMPFLAGS) $(CV_LIBS)

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CXXFILES = $(wildcard $(SDIR)/*.cpp)
CXXFILES = $(notdir $(_CXXFILES))

_OBJ = $(_CXXFILES:.cpp=.o)
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
	@$(PYTHONINTERP) $(TDIR)/plot_speed.py  $(TDIR)/data
	@$(PYTHONINTERP) $(TDIR)/generate_fft.py $(TDIR)/data/signal.txt      $(TDIR)/data/numpy_fft.txt
	@$(PYTHONINTERP) $(TDIR)/compare_fft.py  $(TDIR)/data/transformed.txt $(TDIR)/data/numpy_fft.txt
	@$(PYTHONINTERP) $(TDIR)/plot_same.py    $(TDIR)/data/numpy_fft.txt   $(TDIR)/data/transformed.txt 

$(OBJ) : | subdirs

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(TEST_TARGET): $(TEST_OBJ) $(filter-out $(ODIR)/main.o, $(OBJ))
	$(CXX) -o $@ $^ $(LDFLAGS)	

$(ODIR)/test/%.o: $(TDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	rm -f $(ODIR)/*.o $(TARGET) $(TEST_TARGET) $(LIBRARY_TARGET) $(ODIR)/test/*.o 

.PHONY: clean run subdirs

subdirs: | $(SUBDIRS)

$(SUBDIRS):
	mkdir -p $@
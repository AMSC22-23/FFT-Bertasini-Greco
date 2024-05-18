SHELL := /bin/bash

IDIR = include
SDIR = src
BINDIR = bin
ODIR = obj
OUTDIR = output

FFTIDIR = ../../include
FFTLIBPATH  = ../../lib
FFTLIB = fft

SUBDIRS = $(IDIR) $(SDIR) $(TDIR) $(BINDIR) $(OUTDIR) $(ODIR) $(ODIR)/test

CXX := g++

O_LEVEL = 3

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

CXXFLAGS= -I$(IDIR) -I$(IPYTHON) -I$(INUMPY) -std=c++20 -g -O$(O_LEVEL) -Wall -Wextra $(CXXOMPFLAGS) -I$(FFTIDIR)
LDFLAGS = -L$(LPYTHON) -l$(LIBPYTHON) $(LOMPFLAGS) -L$(FFTLIBPATH) -l$(FFTLIB)

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CXXFILES = $(wildcard $(SDIR)/*.cpp)
CXXFILES = $(notdir $(_CXXFILES))

_OBJ = $(_CXXFILES:.cpp=.o)
OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

TARGET := $(BINDIR)/app

build: subdirs $(TARGET)

run: build $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR)/test/%.o: $(TDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	rm -f $(ODIR)/*.o $(TARGET)

.PHONY: clean run subdirs

subdirs: | $(SUBDIRS)

$(SUBDIRS):
	mkdir -p $@
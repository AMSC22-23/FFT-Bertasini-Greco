SHELL := /bin/bash

IDIR = include
SDIR = src
BINDIR = bin
ODIR = obj
OUTDIR = output

CXX := g++

O_LEVEL = 3

# IPYTHON := /opt/homebrew/opt/python@3.10/Frameworks/Python.framework/Versions/3.10/include/python3.10
# LPYTHON := /opt/homebrew/opt/python@3.10/Frameworks/Python.framework/Versions/3.10/lib
# INUMPY := /opt/homebrew/lib/python3.10/site-packages/numpy/core/include

# find the python include and lib directories
IPYTHON := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
LPYTHON := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
INUMPY  := $(shell python3 -c "import numpy; print(numpy.get_include())")

# remove last part of LPYTHON
LPYTHON := $(shell dirname $(LPYTHON))

# get full python version
PYTHON_VERSION := $(shell python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")

CXXFLAGS= -I$(IDIR) -I$(IPYTHON) -I$(INUMPY) -std=c++20 -g -O$(O_LEVEL) -Wall -Wextra
LDFLAGS = -L$(LPYTHON) -lpython${PYTHON_VERSION}

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CXXFILES = $(wildcard $(SDIR)/*.cpp)
CXXFILES = $(notdir $(_CXXFILES))

_OBJ = $(_CXXFILES:.cpp=.o)
OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

TARGET = $(BINDIR)/app

$(TARGET): $(OBJ) | init
	$(CXX) -o $@ $^ $(LDFLAGS)

all: $(TARGET) run

run: $(TARGET)
	./$(TARGET)

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean run init

clean:
	rm -f $(ODIR)/*.o $(TARGET)

init: | $(BINDIR) $(SDIR) $(IDIR) $(ODIR) $(OUTDIR)

$(ODIR):
	mkdir -p $(ODIR)

$(BINDIR): 
	mkdir -p $(BINDIR)

$(SDIR):
	mkdir -p $(SDIR)

$(IDIR):
	mkdir -p $(IDIR)

$(OUTDIR):
	mkdir -p $(OUTDIR)
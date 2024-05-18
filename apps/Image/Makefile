SHELL := /bin/bash

IDIR = include
SDIR = src
BINDIR = bin
ODIR = obj
OUTDIR = output

FFTIDIR = ../../include
FFTLIBPATH  = ../../lib
FFTLIB = fft

SUBDIRS = $(IDIR) $(SDIR) $(TDIR) $(BINDIR) $(OUTDIR) $(ODIR)

CXX := g++

O_LEVEL = 3

CVWARNINGS_SUPPRESS = -Wno-deprecated-anon-enum-enum-conversion

CV_FLAGS = `pkg-config --cflags opencv4` $(CVWARNINGS_SUPPRESS)
CV_LIBS = `pkg-config --libs opencv4`

# if macos add these flags
ifeq ($(shell uname), Darwin)
	CXXOMPFLAGS := -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
	LOMPFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
else
	CXXOMPFLAGS := -fopenmp
	LOMPFLAGS := -fopenmp
endif

CXXFLAGS= -I$(IDIR) -std=c++20 -g -O$(O_LEVEL) -Wall -Wextra $(CXXOMPFLAGS) $(CV_FLAGS) -I$(FFTIDIR) 
LDFLAGS = $(LOMPFLAGS) $(CV_LIBS) -L$(FFTLIBPATH) -l$(FFTLIB)

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CXXFILES = $(wildcard $(SDIR)/*.cpp)
CXXFILES = $(notdir $(_CXXFILES))

_OBJ = $(_CXXFILES:.cpp=.o)
OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

TARGET := $(BINDIR)/app

build: subdirs $(TARGET)

run: build $(TARGET)
	time ./$(TARGET)

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
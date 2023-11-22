SHELL := /bin/bash

IDIR = include
SDIR = src
BINDIR = bin
ODIR=obj

CXX := g++

O_LEVEL = 3

CXXFLAGS= -I$(IDIR) -std=c++20 -g -O$(O_LEVEL) -Wall -Wextra

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

init: | $(BINDIR) $(SDIR) $(IDIR) $(ODIR)

$(ODIR):
	mkdir -p $(ODIR)

$(BINDIR): 
	mkdir -p $(BINDIR)

$(SDIR):
	mkdir -p $(SDIR)

$(IDIR):
	mkdir -p $(IDIR)
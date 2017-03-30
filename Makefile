CXX ?= g++
CFLAGS = -O3  
LIBS = ./eigen/
all: 
	$(CXX) -std=c++0x $(CFLAGS) -I $(LIBS) run.cpp  -o run

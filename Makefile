CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: preprocess_openmp

preprocess_openmp: preprocess_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(OPENCV_FLAGS)

clean:
	rm -f preprocess_openmp

.PHONY: all clean

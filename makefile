OPENCV = `pkg-config --cflags --libs opencv`


all: 
	g++ --std=c++11 -fopenmp -O3 -I./ main.cpp pm.cpp $(OPENCV) -o pm

	
debug:
	rm -f mm
	g++ -g -Wall main.cpp linalg.h -lpthread -fopenmp -o mm

opt:
	rm -f mm-o
	g++ -O3 -Wall main.cpp linalg.h -lpthread -fopenmp  -o mm-o

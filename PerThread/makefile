all: Duffing.exe

Duffing.exe: Duffing.o
	nvcc -o	Duffing.exe	Duffing.o ParametricODESolver.o

Duffing.o: Duffing.cu
	nvcc -c Duffing.cu			    -O3 -std=c++11
	nvcc -c ParametricODESolver.cu	-O3 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f Duffing.o
	rm -f ParametricODESolver.o
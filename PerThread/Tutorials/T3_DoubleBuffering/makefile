all: DoubleBuffering.exe

DoubleBuffering.exe: DoubleBuffering.o
	nvcc -o	DoubleBuffering.exe	DoubleBuffering.o MassivelyParallel_GPU-ODE_Solver.o

DoubleBuffering.o: DoubleBuffering.cu
	nvcc -c DoubleBuffering.cu			        -O3 -std=c++11
	nvcc -c MassivelyParallel_GPU-ODE_Solver.cu	-O3 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f DoubleBuffering.o
	rm -f MassivelyParallel_GPU-ODE_Solver.o
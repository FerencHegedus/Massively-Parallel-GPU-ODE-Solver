all: Reference.exe

Reference.exe: Reference.o
			   nvcc -o	Reference.exe	Reference.o MassivelyParallel_GPU-ODE_Solver.o

Reference.o: Reference.cu
	nvcc -c Reference.cu			            -O3 -std=c++11
	nvcc -c MassivelyParallel_GPU-ODE_Solver.cu	-O3 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f Reference.o
	rm -f MassivelyParallel_GPU-ODE_Solver.o
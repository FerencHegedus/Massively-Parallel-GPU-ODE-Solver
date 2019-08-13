all: MultiGPUSingleNode.exe

MultiGPUSingleNode.exe: MultiGPUSingleNode.o
	nvcc -o	MultiGPUSingleNode.exe	MultiGPUSingleNode.o MassivelyParallel_GPU-ODE_Solver.o

MultiGPUSingleNode.o: MultiGPUSingleNode.cu
	nvcc -c MultiGPUSingleNode.cu			    -O3 -std=c++11
	nvcc -c MassivelyParallel_GPU-ODE_Solver.cu	-O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f MultiGPUSingleNode.o
	rm -f MassivelyParallel_GPU-ODE_Solver.o
all: ImpactDynamics.exe

ImpactDynamics.exe: ImpactDynamics.o
	nvcc -o	ImpactDynamics.exe	ImpactDynamics.o MassivelyParallel_GPU-ODE_Solver.o

ImpactDynamics.o: ImpactDynamics.cu
	nvcc -c ImpactDynamics.cu			        -O3 -std=c++11
	nvcc -c MassivelyParallel_GPU-ODE_Solver.cu	-O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f ImpactDynamics.o
	rm -f MassivelyParallel_GPU-ODE_Solver.o
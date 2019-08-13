all: QuasiperiodicForcing.exe

QuasiperiodicForcing.exe: QuasiperiodicForcing.o
	nvcc -o	QuasiperiodicForcing.exe	QuasiperiodicForcing.o MassivelyParallel_GPU-ODE_Solver.o

QuasiperiodicForcing.o: QuasiperiodicForcing.cu
	nvcc -c QuasiperiodicForcing.cu			    -O3 -std=c++11
	nvcc -c MassivelyParallel_GPU-ODE_Solver.cu	-O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f QuasiperiodicForcing.o
	rm -f MassivelyParallel_GPU-ODE_Solver.o
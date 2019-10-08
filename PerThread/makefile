INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes

all: Reference.exe

Reference.exe: Reference.o
	nvcc -o	Reference.exe Reference.o

Reference.o: Reference.cu
	nvcc -c $(INCL_DIR) Reference.cu -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

clean:
	rm -f Reference.o
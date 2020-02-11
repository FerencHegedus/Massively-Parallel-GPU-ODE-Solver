INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

all: Poincare.exe

Poincare.exe: Poincare.cu
	nvcc -o	Poincare.exe Poincare.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Poincare.exe
	rm -f *.txt
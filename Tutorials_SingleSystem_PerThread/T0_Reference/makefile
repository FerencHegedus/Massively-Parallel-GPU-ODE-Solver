INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

all: Reference.exe

Reference.exe: Reference.cu
	nvcc -o	Reference.exe Reference.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Reference.exe
	rm -f Duffing.txt
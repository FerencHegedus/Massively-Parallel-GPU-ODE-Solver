INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

all: MultiGPUSingleNode.exe

MultiGPUSingleNode.exe: MultiGPUSingleNode.cu
	nvcc -o	MultiGPUSingleNode.exe MultiGPUSingleNode.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f MultiGPUSingleNode.exe
	rm -f Duffing.txt
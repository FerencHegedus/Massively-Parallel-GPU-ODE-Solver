INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=80

all: BubbleEnsemble.exe

BubbleEnsemble.exe: BubbleEnsemble.cu
	nvcc -o	BubbleEnsemble.exe BubbleEnsemble.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f BubbleEnsemble.exe
	rm -f *.txt
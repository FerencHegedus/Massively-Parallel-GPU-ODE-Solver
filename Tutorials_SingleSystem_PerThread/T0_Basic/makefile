INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -w -maxrregcount=80

all: Logistic.exe

Reference.exe: Logistic.cu
	nvcc -o	Logistic.exe Logistic.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Logistic.exe
	rm -f *.txt

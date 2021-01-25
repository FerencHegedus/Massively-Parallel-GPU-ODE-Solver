INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -w -maxrregcount=80

all: Epidemic.exe

Reference.exe: Epidemic.cu
	nvcc -o	Epidemic.exe Epidemic.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Epidemic.exe
	rm -f *.txt

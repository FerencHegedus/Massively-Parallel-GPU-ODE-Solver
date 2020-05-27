INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128

all: KellerMiksis.exe

KellerMiksis.exe: KellerMiksis.cu
	nvcc -o	KellerMiksis.exe KellerMiksis.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f KellerMiksis.exe
	rm -f KellerMiksis.txt
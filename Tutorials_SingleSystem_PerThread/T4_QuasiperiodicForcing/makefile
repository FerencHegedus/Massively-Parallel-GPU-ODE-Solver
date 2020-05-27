INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128

all: QuasiperiodicForcing.exe

QuasiperiodicForcing.exe: QuasiperiodicForcing.cu
	nvcc -o	QuasiperiodicForcing.exe QuasiperiodicForcing.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f QuasiperiodicForcing.exe
	rm -f *.txt
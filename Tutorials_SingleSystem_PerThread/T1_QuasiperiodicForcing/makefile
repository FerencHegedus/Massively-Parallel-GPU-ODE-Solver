INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

all: QuasiperiodicForcing.exe

QuasiperiodicForcing.exe: QuasiperiodicForcing.cu
	nvcc -o	QuasiperiodicForcing.exe QuasiperiodicForcing.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f QuasiperiodicForcing.exe
	rm -f KellerMiksis_Collapse_PA1_0.50_PA2_0.70.txt
	rm -f KellerMiksis_Collapse_PA1_0.50_PA2_1.20.txt
	rm -f KellerMiksis_Collapse_PA1_1.10_PA2_0.70.txt
	rm -f KellerMiksis_Collapse_PA1_1.10_PA2_1.20.txt
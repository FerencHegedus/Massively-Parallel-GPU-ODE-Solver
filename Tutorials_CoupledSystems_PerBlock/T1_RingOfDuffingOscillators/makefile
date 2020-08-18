INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 --std=c++14 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

all: RingOfDuffingOscillators.exe

RingOfDuffingOscillators.exe: RingOfDuffingOscillators.cu
	nvcc -o	RingOfDuffingOscillators.exe RingOfDuffingOscillators.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f RingOfDuffingOscillators.exe
	rm -f *.txt
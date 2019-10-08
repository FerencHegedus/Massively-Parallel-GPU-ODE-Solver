INCL_DIR = -I$(HOME)/Massively-Parallel-GPU-ODE-Solver/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=64

all: ImpactDynamics.exe

ImpactDynamics.exe: ImpactDynamics.cu
	nvcc -o	ImpactDynamics.exe ImpactDynamics.cu $(INCL_DIR) $(CMPL_OPT)
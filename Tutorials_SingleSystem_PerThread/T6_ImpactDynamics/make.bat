@ECHO off

IF EXIST "ImpactDynamics.exe" del ImpactDynamics.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	ImpactDynamics.exe ImpactDynamics.cu -I%SRC_DIR% %CPL_OPT%

del ImpactDynamics.lib ImpactDynamics.exp

IF EXIST "ImpactDynamics.exe" ECHO Compilation succesful!
IF NOT EXIST "ImpactDynamics.exe" ECHO Compilation failed!
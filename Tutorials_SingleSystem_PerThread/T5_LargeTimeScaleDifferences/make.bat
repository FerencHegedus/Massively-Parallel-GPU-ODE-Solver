@ECHO off

IF EXIST "KellerMiksis.exe" del KellerMiksis.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	KellerMiksis.exe KellerMiksis.cu -I%SRC_DIR% %CPL_OPT%

del KellerMiksis.lib KellerMiksis.exp

IF EXIST "KellerMiksis.exe" ECHO Compilation succesful!
IF NOT EXIST "KellerMiksis.exe" ECHO Compilation failed!
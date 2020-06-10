@ECHO off

IF EXIST "Lorenz.exe" del Lorenz.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	Lorenz.exe Lorenz.cu -I%SRC_DIR% %CPL_OPT%

del Lorenz.lib Lorenz.exp

IF EXIST "Lorenz.exe" ECHO Compilation succesful!
IF NOT EXIST "Lorenz.exe" ECHO Compilation failed!
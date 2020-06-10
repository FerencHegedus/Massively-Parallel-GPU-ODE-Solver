@ECHO off

IF EXIST "Poincare.exe" del Poincare.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	Poincare.exe Poincare.cu -I%SRC_DIR% %CPL_OPT%

del Poincare.lib Poincare.exp

IF EXIST "Poincare.exe" ECHO Compilation succesful!
IF NOT EXIST "Poincare.exe" ECHO Compilation failed!
@ECHO off

IF EXIST "DoubleBuffering.exe" del DoubleBuffering.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	DoubleBuffering.exe DoubleBuffering.cu -I%SRC_DIR% %CPL_OPT%

del DoubleBuffering.lib DoubleBuffering.exp

IF EXIST "DoubleBuffering.exe" ECHO Compilation succesful!
IF NOT EXIST "DoubleBuffering.exe" ECHO Compilation failed!
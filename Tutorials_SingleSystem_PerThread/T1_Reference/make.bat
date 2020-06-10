@ECHO off

IF EXIST "Reference.exe" del Reference.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	Reference.exe Reference.cu -I%SRC_DIR% %CPL_OPT%

del Reference.lib Reference.exp

IF EXIST "Reference.exe" ECHO Compilation succesful!
IF NOT EXIST "Reference.exe" ECHO Compilation failed!
@ECHO off

IF EXIST "MultiGPUSingleNode.exe" del MultiGPUSingleNode.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	MultiGPUSingleNode.exe MultiGPUSingleNode.cu -I%SRC_DIR% %CPL_OPT%

del MultiGPUSingleNode.lib MultiGPUSingleNode.exp

IF EXIST "MultiGPUSingleNode.exe" ECHO Compilation succesful!
IF NOT EXIST "MultiGPUSingleNode.exe" ECHO Compilation failed!
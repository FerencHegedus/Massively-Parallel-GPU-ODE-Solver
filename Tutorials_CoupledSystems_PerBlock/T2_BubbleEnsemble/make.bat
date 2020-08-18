@ECHO off

IF EXIST "BubbleEnsemble.exe" del BubbleEnsemble.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	BubbleEnsemble.exe BubbleEnsemble.cu -I%SRC_DIR% %CPL_OPT%

del BubbleEnsemble.lib BubbleEnsemble.exp

IF EXIST "BubbleEnsemble.exe" ECHO Compilation succesful!
IF NOT EXIST "BubbleEnsemble.exe" ECHO Compilation failed!
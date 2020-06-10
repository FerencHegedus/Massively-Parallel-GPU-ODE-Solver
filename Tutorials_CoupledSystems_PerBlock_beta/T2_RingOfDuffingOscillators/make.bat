@ECHO off

IF EXIST "RingOfDuffingOscillators.exe" del RingOfDuffingOscillators.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	RingOfDuffingOscillators.exe RingOfDuffingOscillators.cu -I%SRC_DIR% %CPL_OPT%

del RingOfDuffingOscillators.lib RingOfDuffingOscillators.exp

IF EXIST "RingOfDuffingOscillators.exe" ECHO Compilation succesful!
IF NOT EXIST "RingOfDuffingOscillators.exe" ECHO Compilation failed!
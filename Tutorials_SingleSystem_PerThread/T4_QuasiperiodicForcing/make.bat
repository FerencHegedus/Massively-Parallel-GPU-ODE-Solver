@ECHO off

IF EXIST "QuasiperiodicForcing.exe" del QuasiperiodicForcing.exe

ECHO Compiling ...

set SRC_DIR=D:\05_Research\ParametricODESolver\Massively-Parallel-GPU-ODE-Solver_TestOnWindows\SourceCodes\
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_52 -lineinfo -w -maxrregcount=80

nvcc -o	QuasiperiodicForcing.exe QuasiperiodicForcing.cu -I%SRC_DIR% %CPL_OPT%

del QuasiperiodicForcing.lib QuasiperiodicForcing.exp

IF EXIST "QuasiperiodicForcing.exe" ECHO Compilation succesful!
IF NOT EXIST "QuasiperiodicForcing.exe" ECHO Compilation failed!
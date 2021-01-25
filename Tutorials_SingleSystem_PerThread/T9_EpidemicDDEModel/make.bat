@ECHO off

IF EXIST "Epidemic.exe" del Logistic.exe

ECHO Compiling ...

set SRC_DIR=C:\Users\nnagy\Documents\Egyetem\HDS\MPGOS\SourceCodes
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_61 -lineinfo -w -maxrregcount=80

nvcc -o	Epidemic.exe Epidemic.cu -I%SRC_DIR% %CPL_OPT%

del Epidemic.lib Epidemic.exp

IF EXIST "Epidemic.exe" ECHO Compilation succesful!
IF NOT EXIST "Epidemic.exe" ECHO Compilation failed!

@ECHO off

IF EXIST "Logistic.exe" del Logistic.exe

ECHO Compiling ...

set SRC_DIR=C:\Users\nnagy\Documents\Egyetem\HDS\MPGOS\SourceCodes
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_61 -lineinfo -w -maxrregcount=80

nvcc -o	Logistic.exe Logistic.cu -I%SRC_DIR% %CPL_OPT%

del Logistic.lib Logistic.exp

IF EXIST "Logistic.exe" ECHO Compilation succesful!
IF NOT EXIST "Logistic.exe" ECHO Compilation failed!

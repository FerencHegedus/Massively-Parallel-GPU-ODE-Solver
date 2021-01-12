@ECHO off

IF EXIST "Reference.exe" del Reference.exe

ECHO Compiling ...

set SRC_DIR=C:\Users\nnagy\Documents\Egyetem\HDS\MPGOS\SourceCodes
set CPL_OPT=-O3 --ptxas-options=-v --gpu-architecture=sm_61 -lineinfo -w -maxrregcount=80

nvcc -o	Reference.exe Reference.cu -I%SRC_DIR% %CPL_OPT%

del Reference.lib Reference.exp

IF EXIST "Reference.exe" ECHO Compilation succesful!
IF NOT EXIST "Reference.exe" ECHO Compilation failed!

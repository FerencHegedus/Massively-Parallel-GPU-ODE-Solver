#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

FileName=$2
LogFileName=$FileName'.log'
ProfFileName=$FileName'.nvprof'

rm -f $LogFileName

echo "--- SUMMARY ---" >> $LogFileName
echo >> $LogFileName

#nvprof --profile-api-trace none ./$1 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName


echo "--- SPECIFIC METRICS AND EVENTS ---" >> $LogFileName
echo >> $LogFileName

sudo nvprof --kernels :::1 --events elapsed_cycles_sm,active_cycles --metrics local_load_throughput ./$1 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName
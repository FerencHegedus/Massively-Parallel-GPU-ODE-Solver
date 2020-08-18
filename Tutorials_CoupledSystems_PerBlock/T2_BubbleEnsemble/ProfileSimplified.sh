#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

FileName=$2
LogFileName=$FileName'.log'
ProfFileName=$FileName'.nvprof'

rm -f $LogFileName

echo "--- SUMMARY ---" >> $LogFileName
echo >> $LogFileName

nvprof --profile-api-trace none ./$1 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName


echo "--- SPECIFIC METRICS AND EVENTS ---" >> $LogFileName
echo >> $LogFileName

nvprof --kernels :::1 --events elapsed_cycles_sm,active_cycles --metrics sm_efficiency,achieved_occupancy,eligible_warps_per_cycle,branch_efficiency,local_load_throughput,local_store_throughput,ipc,issued_ipc,flop_count_dp_add,flop_count_dp_mul,flop_count_dp_fma,inst_integer,inst_control,inst_compute_ld_st,inst_misc,flop_dp_efficiency,l1_shared_utilization,l2_utilization,dram_utilization,ldst_fu_utilization,alu_fu_utilization,stall_pipe_busy,stall_exec_dependency,stall_memory_dependency,stall_inst_fetch,stall_not_selected,stall_memory_throttle,stall_other ./$1 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName
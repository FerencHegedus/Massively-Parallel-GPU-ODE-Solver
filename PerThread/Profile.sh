#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

FileName='Profile'
LogFileName=$FileName'.log'
ProfFileName=$FileName'.nvprof'

rm -f $LogFileName

echo "--- SUMMARY ---" >> $LogFileName
echo >> $LogFileName

nvprof --profile-api-trace none ./Duffing.exe 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName


echo "--- GENERATE FILE FOR VISUAL PROFILER ---" >> $LogFileName
echo >> $LogFileName

nvprof --kernels :::1 --analysis-metrics -o $ProfFileName -f ./Duffing.exe 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName


echo "--- SPECIFIC METRICS AND EVENTS ---" >> $LogFileName
echo >> $LogFileName

#nvprof --kernels :::1 --metrics all --events all ./Duffing.exe 2>>$LogFileName
nvprof --kernels :::1 --events elapsed_cycles_sm,active_cycles --metrics sm_efficiency,achieved_occupancy,eligible_warps_per_cycle,tex_cache_throughput,dram_read_throughput,dram_write_throughput,gst_throughput,gld_throughput,local_load_throughput,local_store_throughput,shared_load_throughput,shared_store_throughput,l2_read_throughput,l2_write_throughput,l2_l1_read_throughput,l2_l1_write_throughput,l2_texture_read_throughput,gld_efficiency,gst_efficiency,shared_efficiency,inst_executed,inst_issued,ipc,issued_ipc,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,flop_dp_efficiency,l1_shared_utilization,l2_utilization,tex_utilization,dram_utilization,sysmem_utilization,ldst_fu_utilization,alu_fu_utilization,cf_fu_utilization,tex_fu_utilization,stall_pipe_busy,stall_exec_dependency,stall_memory_dependency,stall_inst_fetch,stall_texture,stall_not_selected,stall_constant_memory_dependency,stall_memory_throttle,stall_sync,stall_other ./Duffing.exe 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName
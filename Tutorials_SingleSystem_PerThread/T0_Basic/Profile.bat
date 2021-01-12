@ECHO OFF
ECHO Profiling Application...

nvprof --kernels :::1 --events elapsed_cycles_sm,active_cycles --metrics sm_efficiency,achieved_occupancy,eligible_warps_per_cycle,branch_efficiency,tex_cache_throughput,dram_read_throughput,dram_write_throughput,gst_throughput,gld_throughput,local_load_throughput,local_store_throughput,shared_load_throughput,shared_store_throughput,l2_read_throughput,l2_write_throughput,l2_l1_read_throughput,l2_l1_write_throughput,l2_texture_read_throughput,gld_efficiency,gst_efficiency,shared_efficiency,inst_executed,inst_issued,ipc,issued_ipc,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,flop_sp_efficiency,flop_dp_efficiency,l1_shared_utilization,l2_utilization,tex_utilization,dram_utilization,sysmem_utilization,ldst_fu_utilization,alu_fu_utilization,cf_fu_utilization,tex_fu_utilization,stall_pipe_busy,stall_exec_dependency,stall_memory_dependency,stall_inst_fetch,stall_texture,stall_not_selected,stall_constant_memory_dependency,stall_memory_throttle,stall_sync,stall_other --log-file %2 ./%1 >> tmp.log

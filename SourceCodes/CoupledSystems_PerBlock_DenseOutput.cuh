#ifndef COUPLEDSYSTEMs_PERBLOCK_DENSEOUTPUT_H
#define COUPLEDSYSTEMs_PERBLOCK_DENSEOUTPUT_H


template <int NS, int UPS, int UD, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_StoreDenseOutput(\
			int        GlobalSystemID, \
			int        LocalSystemID, \
			int        UnitID, \
			int        s_UpdateDenseOutput, \
			int&       s_DenseOutputIndex, \
			int&       s_NumberOfSkippedStores, \
			Precision& s_DenseOutputActualTime, \
			Precision  UpperTimeDomain, \
			Precision* d_DenseOutputTimeInstances, \
			Precision* d_DenseOutputStates, \
			Precision  s_ActualTime, \
			Precision* r_ActualState, \
			Struct_ThreadConfiguration ThreadConfiguration, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int BlockID = blockIdx.x;
	int GlobalThreadID_Logical;
	int GlobalMemoryID;
	
	int SizeOfActualState = ThreadConfiguration.TotalLogicalThreads*UD;
	
	if ( s_UpdateDenseOutput == 1 )
	{
		for (int i=0; i<UD; i++)
		{
			GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitID;
			GlobalMemoryID         = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads + s_DenseOutputIndex*SizeOfActualState;
			
			d_DenseOutputStates[GlobalMemoryID] = r_ActualState[i];
		}
		
		if ( UnitID == 0 )
		{
			GlobalMemoryID = GlobalSystemID + s_DenseOutputIndex*NS;
			d_DenseOutputTimeInstances[GlobalMemoryID] = s_ActualTime;
			
			s_DenseOutputIndex++;
			s_NumberOfSkippedStores = 0;
			s_DenseOutputActualTime = MPGOS::FMIN(s_ActualTime+SolverOptions.DenseOutputMinimumTimeStep, UpperTimeDomain);
		}
	} else
	{
		if ( UnitID == 0 )
			s_NumberOfSkippedStores++;
	}
}


template <int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_DenseOutputStrageCondition(\
			int       UnitID, \
			int&      s_UpdateDenseOutput, \
			int       s_DenseOutputIndex, \
			int       s_NumberOfSkippedStores, \
			Precision s_DenseOutputActualTime, \
			Precision s_ActualTime, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	if ( UnitID == 0 )
	{
		if ( ( s_DenseOutputIndex < NDO ) && ( s_DenseOutputActualTime < s_ActualTime ) && ( s_NumberOfSkippedStores >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
			s_UpdateDenseOutput = 1;
		else
			s_UpdateDenseOutput = 0;
	}
}

#endif
#ifndef COUPLEDSYSTEMs_PERBLOCK_DENSEOUTPUT_H
#define COUPLEDSYSTEMs_PERBLOCK_DENSEOUTPUT_H


template <int NBL, int NS, int UPS, int UD, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_StoreDenseOutput(\
			int*       s_UpdateDenseOutput, \
			int*       s_DenseOutputIndex, \
			int*       s_NumberOfSkippedStores, \
			Precision* d_DenseOutputTimeInstances, \
			Precision* d_DenseOutputStates, \
			Precision* s_DenseOutputActualTime, \
			Precision* s_ActualTime, \
			Precision  r_ActualState[NBL][UD], \
			Precision  s_TimeDomain[SPB][2], \
			Struct_ThreadConfiguration ThreadConfiguration, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int LocalThreadID_GPU = threadIdx.x;
	int BlockID           = blockIdx.x;
	int GlobalThreadID_Logical;
	int LocalThreadID_Logical;
	int LocalSystemID;
	int GlobalSystemID;
	int GlobalMemoryID;
	int UnitID;
	
	int SizeOfActualState = ThreadConfiguration.TotalLogicalThreads*UD;
	
	for (int BL=0; BL<NBL; BL++)
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( ( LocalSystemID < SPB ) && ( s_UpdateDenseOutput[LocalSystemID] == 1 ) )
		{
			for (int i=0; i<UD; i++)
			{
				GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitID;
				GlobalMemoryID         = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads + s_DenseOutputIndex[LocalSystemID]*SizeOfActualState;
				
				d_DenseOutputStates[GlobalMemoryID] = r_ActualState[BL][i];
			}
			
			if ( UnitID == 0 )
			{
				GlobalMemoryID = GlobalSystemID + s_DenseOutputIndex[LocalSystemID]*NS;
				d_DenseOutputTimeInstances[GlobalMemoryID] = s_ActualTime[LocalSystemID];
			}
		}
		
	}
	__syncthreads();
	
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		LocalSystemID = threadIdx.x + j*blockDim.x;
		
		if ( s_UpdateDenseOutput[LocalSystemID] == 1 )
		{
			s_DenseOutputIndex[LocalSystemID]++;
			s_NumberOfSkippedStores[LocalSystemID] = 0;
			s_DenseOutputActualTime[LocalSystemID] = MPGOS::FMIN(s_ActualTime[LocalSystemID]+SolverOptions.DenseOutputMinimumTimeStep, s_TimeDomain[LocalSystemID][1]);
		} else
			s_NumberOfSkippedStores[LocalSystemID]++;
	}
	__syncthreads();
}


template <int NBL, int UPS, int SPB, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_DenseOutputStrageCondition(\
			int*       s_EndTimeDomainReached, \
			int*       s_UpdateStep, \
			int*       s_UpdateDenseOutput, \
			int*       s_DenseOutputIndex, \
			int*       s_NumberOfSkippedStores, \
			Precision* s_DenseOutputActualTime, \
			Precision* s_ActualTime, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int LocalThreadID_GPU = threadIdx.x;
	int LocalThreadID_Logical;
	int LocalSystemID;
	int UnitID;
	
	for (int BL=0; BL<NBL; BL++)
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( ( LocalSystemID < SPB ) && ( UnitID == 0 ) )
		{
			if ( s_UpdateStep[LocalSystemID] == 1 )
			{
				if ( ( s_DenseOutputIndex[LocalSystemID] < NDO ) && ( s_DenseOutputActualTime[LocalSystemID] < s_ActualTime[LocalSystemID] ) && ( s_NumberOfSkippedStores[LocalSystemID] >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
					s_UpdateDenseOutput[LocalSystemID] = 1;
				else
					s_UpdateDenseOutput[LocalSystemID] = 0;
				
				if ( s_EndTimeDomainReached[LocalSystemID] == 1 )
					s_UpdateDenseOutput[LocalSystemID] = 1;
			} else
				s_UpdateDenseOutput[LocalSystemID] = 0;
		}
	}
	__syncthreads();
}

#endif
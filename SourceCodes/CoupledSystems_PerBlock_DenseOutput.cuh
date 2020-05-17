#ifndef COUPLEDSYSTEMs_PERBLOCK_DENSEOUTPUT_H
#define COUPLEDSYSTEMs_PERBLOCK_DENSEOUTPUT_H


// MSMBL ----------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_StoreDenseOutput(\
			int*       s_UpdateStep, \
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
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
		{
			if ( s_UpdateDenseOutput[lsid] == 1 )
			{
				s_DenseOutputIndex[lsid]++;
				s_NumberOfSkippedStores[lsid] = 0;
				s_DenseOutputActualTime[lsid] = MPGOS::FMIN(s_ActualTime[lsid]+SolverOptions.DenseOutputMinimumTimeStep, s_TimeDomain[lsid][1]);
			}
			
			if ( ( s_UpdateDenseOutput[lsid] == 0 ) && ( s_UpdateStep[lsid] == 1 ) )
				s_NumberOfSkippedStores[lsid]++;
		}
	}
	__syncthreads();
}


template <int NBL, int UPS, int SPB, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_DenseOutputStorageCondition(\
			int*       s_EndTimeDomainReached, \
			int*       s_UserDefinedTermination, \
			int*       s_UpdateStep, \
			int*       s_UpdateDenseOutput, \
			int*       s_DenseOutputIndex, \
			int*       s_NumberOfSkippedStores, \
			Precision* s_DenseOutputActualTime, \
			Precision* s_ActualTime, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
		{
			if ( s_UpdateStep[lsid] == 1 )
			{
				if ( ( s_DenseOutputIndex[lsid] < NDO ) && ( s_DenseOutputActualTime[lsid] < s_ActualTime[lsid] ) && ( s_NumberOfSkippedStores[lsid] >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
					s_UpdateDenseOutput[lsid] = 1;
				else
					s_UpdateDenseOutput[lsid] = 0;
				
				if ( ( s_DenseOutputIndex[lsid] < NDO ) && ( ( s_EndTimeDomainReached[lsid] == 1 ) || ( s_UserDefinedTermination[lsid] == 1 ) ) )
					s_UpdateDenseOutput[lsid] = 1;
			} else
				s_UpdateDenseOutput[lsid] = 0;
		}
	}
	__syncthreads();
}


// SSMBL ----------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_StoreDenseOutput(\
			int        s_UpdateStep, \
			int        s_UpdateDenseOutput, \
			int&       s_DenseOutputIndex, \
			int&       s_NumberOfSkippedStores, \
			Precision* d_DenseOutputTimeInstances, \
			Precision* d_DenseOutputStates, \
			Precision& s_DenseOutputActualTime, \
			Precision  s_ActualTime, \
			Precision  r_ActualState[NBL][UD], \
			Precision  s_TimeDomain[2], \
			Struct_ThreadConfiguration ThreadConfiguration, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int LocalThreadID_GPU = threadIdx.x;
	int GlobalSystemID    = blockIdx.x;
	int GlobalThreadID_Logical;
	int UnitID;
	int GlobalMemoryID;
	
	int SizeOfActualState = ThreadConfiguration.TotalLogicalThreads*UD;
	
	for (int BL=0; BL<NBL; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		GlobalThreadID_Logical = GlobalSystemID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + UnitID;
		
		if ( ( UnitID < UPS ) && ( s_UpdateDenseOutput == 1 ) )
		{
			for (int i=0; i<UD; i++)
			{
				GlobalMemoryID = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads + s_DenseOutputIndex*SizeOfActualState;
				
				d_DenseOutputStates[GlobalMemoryID] = r_ActualState[BL][i];
			}
			
			if ( UnitID == 0 )
			{
				GlobalMemoryID = GlobalSystemID + s_DenseOutputIndex*NS;
				d_DenseOutputTimeInstances[GlobalMemoryID] = s_ActualTime;
			}
		}
		
	}
	__syncthreads();
	
	if ( threadIdx.x == 0 )
	{
		if ( s_UpdateDenseOutput == 1 )
		{
			s_DenseOutputIndex++;
			s_NumberOfSkippedStores = 0;
			s_DenseOutputActualTime = MPGOS::FMIN(s_ActualTime+SolverOptions.DenseOutputMinimumTimeStep, s_TimeDomain[1]);
		}
		
		if ( ( s_UpdateDenseOutput == 0 ) && ( s_UpdateStep == 1 ) )
			s_NumberOfSkippedStores++;
	}
	__syncthreads();
}


template <int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_DenseOutputStorageCondition(\
			int       s_EndTimeDomainReached, \
			int       s_UserDefinedTermination, \
			int       s_UpdateStep, \
			int&      s_UpdateDenseOutput, \
			int       s_DenseOutputIndex, \
			int       s_NumberOfSkippedStores, \
			Precision s_DenseOutputActualTime, \
			Precision s_ActualTime, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	if ( threadIdx.x == 0 )
	{
		if ( s_UpdateStep == 1 )
		{
			if ( ( s_DenseOutputIndex < NDO ) && ( s_DenseOutputActualTime < s_ActualTime ) && ( s_NumberOfSkippedStores >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
				s_UpdateDenseOutput = 1;
			else
				s_UpdateDenseOutput = 0;
			
			if ( ( s_DenseOutputIndex < NDO ) && ( ( s_EndTimeDomainReached == 1 ) || ( s_UserDefinedTermination == 1 ) ) )
				s_UpdateDenseOutput = 1;
		} else
			s_UpdateDenseOutput = 0;
	}
	__syncthreads();
}


// MSSBL ----------------------------------------------------------------------
template <int NS, int UPS, int UD, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_StoreDenseOutput(\
			const int  BlockID, \
			const int  LocalSystemID, \
			const int  UnitID, \
			const int  GlobalSystemID, \
			int*       s_UpdateStep, \
			int*       s_UpdateDenseOutput, \
			int*       s_DenseOutputIndex, \
			int*       s_NumberOfSkippedStores, \
			Precision* d_DenseOutputTimeInstances, \
			Precision* d_DenseOutputStates, \
			Precision* s_DenseOutputActualTime, \
			Precision* s_ActualTime, \
			Precision  r_ActualState[UD], \
			Precision  s_TimeDomain[SPB][2], \
			Struct_ThreadConfiguration ThreadConfiguration, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int GlobalThreadID_Logical;
	int GlobalMemoryID;
	
	int SizeOfActualState = ThreadConfiguration.TotalLogicalThreads*UD;
	
	if ( ( LocalSystemID < SPB ) && ( s_UpdateDenseOutput[LocalSystemID] == 1 ) )
	{
		for (int i=0; i<UD; i++)
		{
			GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitID;
			GlobalMemoryID         = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads + s_DenseOutputIndex[LocalSystemID]*SizeOfActualState;
			
			d_DenseOutputStates[GlobalMemoryID] = r_ActualState[i];
		}
		
		if ( UnitID == 0 )
		{
			GlobalMemoryID = GlobalSystemID + s_DenseOutputIndex[LocalSystemID]*NS;
			d_DenseOutputTimeInstances[GlobalMemoryID] = s_ActualTime[LocalSystemID];
		}
	}
	__syncthreads();
	
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
		{
			if ( s_UpdateDenseOutput[lsid] == 1 )
			{
				s_DenseOutputIndex[lsid]++;
				s_NumberOfSkippedStores[lsid] = 0;
				s_DenseOutputActualTime[lsid] = MPGOS::FMIN(s_ActualTime[lsid]+SolverOptions.DenseOutputMinimumTimeStep, s_TimeDomain[lsid][1]);
			}
			
			if ( ( s_UpdateDenseOutput[lsid] == 0 ) && ( s_UpdateStep[lsid] == 1 ) )
				s_NumberOfSkippedStores[lsid]++;
		}
	}
	__syncthreads();
}


template <int UPS, int SPB, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_DenseOutputStorageCondition(\
			int*       s_EndTimeDomainReached, \
			int*       s_UserDefinedTermination, \
			int*       s_UpdateStep, \
			int*       s_UpdateDenseOutput, \
			int*       s_DenseOutputIndex, \
			int*       s_NumberOfSkippedStores, \
			Precision* s_DenseOutputActualTime, \
			Precision* s_ActualTime, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid  = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
		{
			if ( s_UpdateStep[lsid] == 1 )
			{
				if ( ( s_DenseOutputIndex[lsid] < NDO ) && ( s_DenseOutputActualTime[lsid] < s_ActualTime[lsid] ) && ( s_NumberOfSkippedStores[lsid] >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
					s_UpdateDenseOutput[lsid] = 1;
				else
					s_UpdateDenseOutput[lsid] = 0;
				
				if ( ( s_DenseOutputIndex[lsid] < NDO ) && ( ( s_EndTimeDomainReached[lsid] == 1 ) || ( s_UserDefinedTermination[lsid] == 1 ) ) )
					s_UpdateDenseOutput[lsid] = 1;
			} else
				s_UpdateDenseOutput[lsid] = 0;
		}
	}
	__syncthreads();
}


// SSSBL ----------------------------------------------------------------------
template <int NS, int UPS, int UD, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_StoreDenseOutput(\
			const int  LocalThreadID, \
			const int  GlobalThreadID, \
			const int  GlobalSystemID, \
			int        s_UpdateStep, \
			int        s_UpdateDenseOutput, \
			int&       s_DenseOutputIndex, \
			int&       s_NumberOfSkippedStores, \
			Precision* d_DenseOutputTimeInstances, \
			Precision* d_DenseOutputStates, \
			Precision& s_DenseOutputActualTime, \
			Precision  s_ActualTime, \
			Precision  r_ActualState[UD], \
			Precision  s_TimeDomain[2], \
			Struct_ThreadConfiguration ThreadConfiguration, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int GlobalMemoryID;
	
	int SizeOfActualState = ThreadConfiguration.TotalLogicalThreads*UD;
	
	if ( ( LocalThreadID < UPS ) && ( s_UpdateDenseOutput == 1 ) )
	{
		for (int i=0; i<UD; i++)
		{
			GlobalMemoryID = GlobalThreadID + i*ThreadConfiguration.TotalLogicalThreads + s_DenseOutputIndex*SizeOfActualState;
			
			d_DenseOutputStates[GlobalMemoryID] = r_ActualState[i];
		}
		
		if ( LocalThreadID == 0 )
		{
			GlobalMemoryID = GlobalSystemID + s_DenseOutputIndex*NS;
			d_DenseOutputTimeInstances[GlobalMemoryID] = s_ActualTime;
		}
	}
	__syncthreads();
	
	if ( threadIdx.x == 0 )
	{
		if ( s_UpdateDenseOutput == 1 )
		{
			s_DenseOutputIndex++;
			s_NumberOfSkippedStores = 0;
			s_DenseOutputActualTime = MPGOS::FMIN(s_ActualTime+SolverOptions.DenseOutputMinimumTimeStep, s_TimeDomain[1]);
		}
		
		if ( ( s_UpdateDenseOutput == 0 ) && ( s_UpdateStep == 1 ) )
			s_NumberOfSkippedStores++;
	}
	__syncthreads();
}


template <int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_DenseOutputStorageCondition(\
			int       s_EndTimeDomainReached, \
			int       s_UserDefinedTermination, \
			int       s_UpdateStep, \
			int&      s_UpdateDenseOutput, \
			int       s_DenseOutputIndex, \
			int       s_NumberOfSkippedStores, \
			Precision s_DenseOutputActualTime, \
			Precision s_ActualTime, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	if ( threadIdx.x == 0 )
	{
		if ( s_UpdateStep == 1 )
		{
			if ( ( s_DenseOutputIndex < NDO ) && ( s_DenseOutputActualTime < s_ActualTime ) && ( s_NumberOfSkippedStores >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
				s_UpdateDenseOutput = 1;
			else
				s_UpdateDenseOutput = 0;
			
			if ( ( s_DenseOutputIndex < NDO ) && ( ( s_EndTimeDomainReached == 1 ) || ( s_UserDefinedTermination == 1 ) ) )
				s_UpdateDenseOutput = 1;
		} else
			s_UpdateDenseOutput = 0;
	}
	__syncthreads();
}

#endif
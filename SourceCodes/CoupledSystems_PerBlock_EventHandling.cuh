#ifndef COUPLEDSYSTEMS_PERBLOCK_EVENTHANDLING_H
#define COUPLEDSYSTEMS_PERBLOCK_EVENTHANDLING_H


template <int NBL, int UPS, int SPB, int NE, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_EventTimeStepControl(\
			Precision* s_TimeStep, \
			Precision* s_NewTimeStep, \
			int*       s_TerminateSystemScope, \
			int*       s_UpdateStep, \
			Precision  r_ActualEventValue[(NE==0?1:NBL)][(NE==0?1:NE)], \
			Precision  r_NextEventValue[(NE==0?1:NBL)][(NE==0?1:NE)], \
			Precision* s_EventTolerance, \
			int*       s_EventDirection, \
			Precision  MinimumTimeStep)
{
	int LocalThreadID_GPU = threadIdx.x;
	int BlockID           = blockIdx.x;
	int LocalThreadID_Logical;
	int LocalSystemID;
	int GlobalSystemID;
	
	__shared__ Precision s_EventTimeStep[SPB];
	__shared__ int       s_IsCorrected[SPB];
	
	// Event time step initialisation
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
		{
			s_EventTimeStep[lsid] = s_TimeStep[lsid];
			s_IsCorrected[lsid]   = 0;
		}
	}
	__syncthreads();
	
	// Event time step correction
	for (int BL=0; BL<NBL; BL++)
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		
		if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
		{
			for (int i=0; i<NE; i++)
			{
				if ( ( ( r_ActualEventValue[BL][i] >  s_EventTolerance[i] ) && ( r_NextEventValue[BL][i] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
				     ( ( r_ActualEventValue[BL][i] < -s_EventTolerance[i] ) && ( r_NextEventValue[BL][i] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
				{
					MPGOS::atomicFMIN(&(s_EventTimeStep[LocalSystemID]), -r_ActualEventValue[BL][i] / (r_NextEventValue[BL][i]-r_ActualEventValue[BL][i]) * s_TimeStep[LocalSystemID]);
					atomicMax(&(s_IsCorrected[LocalSystemID]), 1);
				}
			}
		}
	}
	__syncthreads();
	
	// Corrected time step and modified update
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		int gsid = lsid + blockIdx.x*SPB;
		
		if ( ( lsid < SPB ) && ( s_IsCorrected[lsid] == 1 ) )
		{
			if ( s_EventTimeStep[lsid] < MinimumTimeStep )
			{
				printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, (global system id: %d)\n", gsid);
			} else
			{
				s_NewTimeStep[lsid] = s_EventTimeStep[lsid];
				s_UpdateStep[lsid]  = 0;
			}
		}
	}
	__syncthreads();
}


template <int NBL, int UPS, int NE, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_EventTimeStepControl(\
			Precision  s_TimeStep, \
			Precision& s_NewTimeStep, \
			int        s_TerminateSystemScope, \
			int&       s_UpdateStep, \
			Precision  r_ActualEventValue[(NE==0?1:NBL)][(NE==0?1:NE)], \
			Precision  r_NextEventValue[(NE==0?1:NBL)][(NE==0?1:NE)], \
			Precision* s_EventTolerance, \
			int*       s_EventDirection, \
			Precision  MinimumTimeStep)
{
	const int LocalThreadID_GPU = threadIdx.x;
	const int GlobalSystemID    = blockIdx.x;
	
	int UnitID;
	
	
	__shared__ Precision s_EventTimeStep;
	__shared__ int       s_IsCorrected;
	
	// Event time step initialisation
	if ( threadIdx.x == 0 )
	{
		s_EventTimeStep = s_TimeStep;
		s_IsCorrected   = 0;
	}
	__syncthreads();
	
	// Event time step correction
	for (int BL=0; BL<NBL; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( ( UnitID < UPS ) && ( s_UpdateStep == 1 ) && ( s_TerminateSystemScope == 0 ) )
		{
			for (int i=0; i<NE; i++)
			{
				if ( ( ( r_ActualEventValue[BL][i] >  s_EventTolerance[i] ) && ( r_NextEventValue[BL][i] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
				     ( ( r_ActualEventValue[BL][i] < -s_EventTolerance[i] ) && ( r_NextEventValue[BL][i] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
				{
					MPGOS::atomicFMIN(&s_EventTimeStep, -r_ActualEventValue[BL][i] / (r_NextEventValue[BL][i]-r_ActualEventValue[BL][i]) * s_TimeStep);
					atomicMax(&s_IsCorrected, 1);
				}
			}
		}
	}
	__syncthreads();
	
	// Corrected time step and modified update
	if ( ( threadIdx.x == 0 ) && ( s_IsCorrected == 1 ) )
	{
		if ( s_EventTimeStep < MinimumTimeStep )
		{
			printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, (global system id: %d)\n", GlobalSystemID);
		} else
		{
			s_NewTimeStep = s_EventTimeStep;
			s_UpdateStep  = 0;
		}
	}
	__syncthreads();
}


template <int UPS, int SPB, int NE, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_EventTimeStepControl(\
			const int  LocalSystemID, \
			Precision* s_TimeStep, \
			Precision* s_NewTimeStep, \
			int*       s_TerminateSystemScope, \
			int*       s_UpdateStep, \
			Precision  r_ActualEventValue[(NE==0?1:NE)], \
			Precision  r_NextEventValue[(NE==0?1:NE)], \
			Precision* s_EventTolerance, \
			int*       s_EventDirection, \
			Precision  MinimumTimeStep)
{
	__shared__ Precision s_EventTimeStep[SPB];
	__shared__ int       s_IsCorrected[SPB];
	
	// Event time step initialisation
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
		{
			s_EventTimeStep[lsid] = s_TimeStep[lsid];
			s_IsCorrected[lsid]   = 0;
		}
	}
	__syncthreads();
	
	// Event time step correction
	if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
	{
		for (int i=0; i<NE; i++)
		{
			if ( ( ( r_ActualEventValue[i] >  s_EventTolerance[i] ) && ( r_NextEventValue[i] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
			     ( ( r_ActualEventValue[i] < -s_EventTolerance[i] ) && ( r_NextEventValue[i] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
			{
				MPGOS::atomicFMIN(&(s_EventTimeStep[LocalSystemID]), -r_ActualEventValue[i] / (r_NextEventValue[i]-r_ActualEventValue[i]) * s_TimeStep[LocalSystemID]);
				atomicMax(&(s_IsCorrected[LocalSystemID]), 1);
			}
		}
	}
	__syncthreads();
	
	// Corrected time step and modified update
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		int gsid = lsid + blockIdx.x*SPB;
		
		if ( ( lsid < SPB ) && ( s_IsCorrected[lsid] == 1 ) )
		{
			if ( s_EventTimeStep[lsid] < MinimumTimeStep )
			{
				printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, (global system id: %d)\n", gsid);
			} else
			{
				s_NewTimeStep[lsid] = s_EventTimeStep[lsid];
				s_UpdateStep[lsid]  = 0;
			}
		}
	}
	__syncthreads();
}


template <int UPS, int NE, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_EventTimeStepControl(\
			const int  GlobalSystemID, \
			const int  LocalThreadID, \
			Precision  s_TimeStep, \
			Precision& s_NewTimeStep, \
			int        s_TerminateSystemScope, \
			int&       s_UpdateStep, \
			Precision  r_ActualEventValue[(NE==0?1:NE)], \
			Precision  r_NextEventValue[(NE==0?1:NE)], \
			Precision* s_EventTolerance, \
			int*       s_EventDirection, \
			Precision  MinimumTimeStep)
{
	__shared__ Precision s_EventTimeStep;
	__shared__ int       s_IsCorrected;
	
	// Event time step initialisation
	if ( threadIdx.x == 0 )
	{
		s_EventTimeStep = s_TimeStep;
		s_IsCorrected   = 0;
	}
	__syncthreads();
	
	// Event time step correction
	if ( ( LocalThreadID < UPS ) && ( s_UpdateStep == 1 ) && ( s_TerminateSystemScope == 0 ) )
	{
		for (int i=0; i<NE; i++)
		{
			if ( ( ( r_ActualEventValue[i] >  s_EventTolerance[i] ) && ( r_NextEventValue[i] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
			     ( ( r_ActualEventValue[i] < -s_EventTolerance[i] ) && ( r_NextEventValue[i] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
			{
				MPGOS::atomicFMIN(&s_EventTimeStep, -r_ActualEventValue[i] / (r_NextEventValue[i]-r_ActualEventValue[i]) * s_TimeStep);
				atomicMax(&s_IsCorrected, 1);
			}
		}
	}
	__syncthreads();
	
	// Corrected time step and modified update
	if ( ( threadIdx.x == 0 ) && ( s_IsCorrected == 1 ) )
	{
		if ( s_EventTimeStep < MinimumTimeStep )
		{
			printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, (global system id: %d)\n", GlobalSystemID);
		} else
		{
			s_NewTimeStep = s_EventTimeStep;
			s_UpdateStep  = 0;
		}
	}
	__syncthreads();
}

#endif
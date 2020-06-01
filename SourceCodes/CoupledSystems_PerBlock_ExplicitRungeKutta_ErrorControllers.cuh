#ifndef COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_ERRORCONTROLLERS_H
#define COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_ERRORCONTROLLERS_H


template <int NBL, int UPS, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_ErrorController_RK4(\
			int*       s_IsFinite, \
			Precision* s_NewTimeStep, \
			Precision  InitialTimeStep, \
			int*       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int*       s_UpdateStep)
{
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		int gsid = lsid + blockIdx.x*SPB;
		
		if ( ( lsid < SPB ) && ( s_TerminateSystemScope[lsid] == 0 ) )
		{
			if ( s_IsFinite[lsid] == 0 )
			{
				printf("Error: State is not a finite number. Try to use smaller step size. (global system id: %d)\n", gsid);
				
				s_TerminateSystemScope[lsid] = 1;
				atomicAdd(&s_TerminatedSystemsPerBlock, 1);
				
				s_UpdateStep[lsid] = 0;
			}
			
			s_NewTimeStep[lsid] = InitialTimeStep;
		}
	}
	__syncthreads();
}


template <int NBL, int UPS, int UD, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_ErrorController_RKCK45(\
			int*       s_IsFinite, \
			Precision* s_TimeStep, \
			Precision* s_NewTimeStep, \
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision  r_Error[NBL][UD], \
			Precision* s_RelativeTolerance, \
			Precision* s_AbsoluteTolerance, \
			int*       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int*       s_UpdateStep, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	int LocalThreadID_GPU = threadIdx.x;
	int BlockID           = blockIdx.x;
	int LocalThreadID_Logical;
	int LocalSystemID;
	int GlobalSystemID;
	
	__shared__ Precision s_RelativeError[SPB];
	__shared__ Precision s_TimeStepMultiplicator[SPB];
	
	Precision r_ErrorTolerance;
	Precision r_RelativeError;
	int       r_UpdateStep;
	
	// Relative error initialisation
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
			s_RelativeError[lsid] = 1e30;
	}
	__syncthreads();
	
	// Acceptance and calculation of relative error (reduction tasks)
	for (int BL=0; BL<NBL; BL++)
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		
		if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
		{
			r_RelativeError = 1e30;
			r_UpdateStep    = 1;
			
			for (int i=0; i<UD; i++)
			{
				r_ErrorTolerance = MPGOS::FMAX( s_RelativeTolerance[i]*MPGOS::FMAX( MPGOS::FABS(r_NextState[BL][i]), MPGOS::FABS(r_ActualState[BL][i]) ), s_AbsoluteTolerance[i] );
				r_UpdateStep     = r_UpdateStep & ( r_Error[BL][i] < r_ErrorTolerance );
				r_RelativeError  = MPGOS::FMIN( r_RelativeError, r_ErrorTolerance / r_Error[BL][i] );
			}
			
			atomicAnd(&(s_UpdateStep[LocalSystemID]), r_UpdateStep);
			MPGOS::atomicFMIN(&(s_RelativeError[LocalSystemID]), r_RelativeError);
		}
	}
	__syncthreads();
	
	// New time step (with bound checks)
	Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		int gsid = lsid + blockIdx.x*SPB;
		
		if ( ( lsid < SPB ) && ( s_TerminateSystemScope[lsid] == 0 ) )
		{
			// Base time step multiplicator
			if ( s_UpdateStep[lsid] == 1 )
				s_TimeStepMultiplicator[lsid] = static_cast<Precision>(0.8) * pow(s_RelativeError[lsid], static_cast<Precision>(1.0/5.0) ); // 1.0/5.0
			else
				s_TimeStepMultiplicator[lsid] = static_cast<Precision>(0.8) * pow(s_RelativeError[lsid], static_cast<Precision>(1.0/4.0) ); // 1.0/4.0
			
			if ( isfinite(s_TimeStepMultiplicator[lsid]) == 0 )
				s_IsFinite[lsid] = 0;
			
			// Check finiteness
			if ( s_IsFinite[lsid] == 0 )
			{
				s_TimeStepMultiplicator[lsid] = SolverOptions.TimeStepShrinkLimit;
				s_UpdateStep[lsid] = 0;
				
				if ( s_TimeStep[lsid] < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
				{
					printf("Error: State is not a finite number, minimum step size reached. Try to use less stringent tolerances. (global system id: %d)\n", gsid);
					s_TerminateSystemScope[lsid] = 1;
					atomicAdd(&s_TerminatedSystemsPerBlock, 1);
				}
			} else
			{
				if ( s_TimeStep[lsid] < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
				{
					printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed! (global system id: %d, time step: %+6.5e, TSM: %+6.3e \n", gsid, SolverOptions.MinimumTimeStep, s_TimeStepMultiplicator[lsid]);
					s_UpdateStep[lsid] = 1;
				}
			}
			
			// Time step and its growth limits
			s_TimeStepMultiplicator[lsid] = MPGOS::FMIN(s_TimeStepMultiplicator[lsid], SolverOptions.TimeStepGrowLimit);
			s_TimeStepMultiplicator[lsid] = MPGOS::FMAX(s_TimeStepMultiplicator[lsid], SolverOptions.TimeStepShrinkLimit);
			
			s_NewTimeStep[lsid] = s_TimeStep[lsid] * s_TimeStepMultiplicator[lsid];
			
			s_NewTimeStep[lsid] = MPGOS::FMIN(s_NewTimeStep[lsid], SolverOptions.MaximumTimeStep);
			s_NewTimeStep[lsid] = MPGOS::FMAX(s_NewTimeStep[lsid], SolverOptions.MinimumTimeStep);
		}
	}
	__syncthreads();
}


template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_ErrorController_RK4(\
			int        s_IsFinite, \
			Precision& s_NewTimeStep, \
			Precision  InitialTimeStep, \
			int&       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int&       s_UpdateStep)
{
	const int GlobalSystemID = blockIdx.x;
	
	if ( ( threadIdx.x == 0 ) && ( s_TerminateSystemScope == 0 ) )
	{
		if ( s_IsFinite == 0 )
		{
			printf("Error: State is not a finite number. Try to use smaller step size. (global system id: %d)\n", GlobalSystemID);
			
			s_TerminateSystemScope = 1;
			atomicAdd(&s_TerminatedSystemsPerBlock, 1);
			
			s_UpdateStep = 0;
		}
		
		s_NewTimeStep = InitialTimeStep;
	}
	__syncthreads();
}


template <int NBL, int UPS, int UD, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_ErrorController_RKCK45(\
			int&       s_IsFinite, \
			Precision  s_TimeStep, \
			Precision& s_NewTimeStep, \
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision  r_Error[NBL][UD], \
			Precision* s_RelativeTolerance, \
			Precision* s_AbsoluteTolerance, \
			int&       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int&       s_UpdateStep, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	const int LocalThreadID_GPU = threadIdx.x;
	const int GlobalSystemID    = blockIdx.x;
	
	int UnitID;
	
	
	__shared__ Precision s_RelativeError;
	__shared__ Precision s_TimeStepMultiplicator;
	
	Precision r_ErrorTolerance;
	Precision r_RelativeError;
	int       r_UpdateStep;
	
	// Relative error initialisation
	if ( threadIdx.x == 0 )
		s_RelativeError = 1e30;
	__syncthreads();
	
	// Acceptance and calculation of relative error (reduction tasks)
	for (int BL=0; BL<NBL; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( ( UnitID < UPS ) && ( s_TerminateSystemScope == 0 ) )
		{
			r_RelativeError = 1e30;
			r_UpdateStep    = 1;
			
			for (int i=0; i<UD; i++)
			{
				r_ErrorTolerance = MPGOS::FMAX( s_RelativeTolerance[i]*MPGOS::FMAX( MPGOS::FABS(r_NextState[BL][i]), MPGOS::FABS(r_ActualState[BL][i]) ), s_AbsoluteTolerance[i] );
				r_UpdateStep     = r_UpdateStep & ( r_Error[BL][i] < r_ErrorTolerance );
				r_RelativeError  = MPGOS::FMIN( r_RelativeError, r_ErrorTolerance / r_Error[BL][i] );
			}
			
			atomicAnd(&s_UpdateStep, r_UpdateStep);
			MPGOS::atomicFMIN(&s_RelativeError, r_RelativeError);
		}
	}
	__syncthreads();
	
	if ( ( threadIdx.x == 0 ) && ( s_TerminateSystemScope == 0 ) )
	{
		// Base time step multiplicator
		if ( s_UpdateStep == 1 )
			s_TimeStepMultiplicator = static_cast<Precision>(0.8) * pow(s_RelativeError, static_cast<Precision>(1.0/5.0) ); // 1.0/5.0
		else
			s_TimeStepMultiplicator = static_cast<Precision>(0.8) * pow(s_RelativeError, static_cast<Precision>(1.0/4.0) ); // 1.0/4.0
		
		if ( isfinite(s_TimeStepMultiplicator) == 0 )
			s_IsFinite = 0;
		
		// Check finiteness
		if ( s_IsFinite == 0 )
		{
			s_TimeStepMultiplicator = SolverOptions.TimeStepShrinkLimit;
			s_UpdateStep = 0;
			
			if ( s_TimeStep < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
			{
				printf("Error: State is not a finite number, minimum step size reached. Try to use less stringent tolerances. (global system id: %d)\n", GlobalSystemID);
				s_TerminateSystemScope = 1;
				atomicAdd(&s_TerminatedSystemsPerBlock, 1);
			}
		} else
		{
			if ( s_TimeStep < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
			{
				printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed! (global system id: %d, time step: %+6.5e, TSM: %+6.3e \n", GlobalSystemID, SolverOptions.MinimumTimeStep, s_TimeStepMultiplicator);
				s_UpdateStep = 1;
			}
		}
		
		// Time step and its growth limits
		s_TimeStepMultiplicator = MPGOS::FMIN(s_TimeStepMultiplicator, SolverOptions.TimeStepGrowLimit);
		s_TimeStepMultiplicator = MPGOS::FMAX(s_TimeStepMultiplicator, SolverOptions.TimeStepShrinkLimit);
		
		s_NewTimeStep = s_TimeStep * s_TimeStepMultiplicator;
		
		s_NewTimeStep = MPGOS::FMIN(s_NewTimeStep, SolverOptions.MaximumTimeStep);
		s_NewTimeStep = MPGOS::FMAX(s_NewTimeStep, SolverOptions.MinimumTimeStep);
	}
	__syncthreads();
}


template <int UPS, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_ErrorController_RK4(\
			int*       s_IsFinite, \
			Precision* s_NewTimeStep, \
			Precision  InitialTimeStep, \
			int*       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int*       s_UpdateStep)
{
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		int gsid = lsid + blockIdx.x*SPB;
		
		if ( ( lsid < SPB ) && ( s_TerminateSystemScope[lsid] == 0 ) )
		{
			if ( s_IsFinite[lsid] == 0 )
			{
				printf("Error: State is not a finite number. Try to use smaller step size. (global system id: %d)\n", gsid);
				
				s_TerminateSystemScope[lsid] = 1;
				atomicAdd(&s_TerminatedSystemsPerBlock, 1);
				
				s_UpdateStep[lsid] = 0;
			}
			
			s_NewTimeStep[lsid] = InitialTimeStep;
		}
	}
	__syncthreads();
}


template <int UPS, int UD, int SPB, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_ErrorController_RKCK45(\
			const int  LocalSystemID, \
			int*       s_IsFinite, \
			Precision* s_TimeStep, \
			Precision* s_NewTimeStep, \
			Precision  r_ActualState[UD], \
			Precision  r_NextState[UD], \
			Precision  r_Error[UD], \
			Precision* s_RelativeTolerance, \
			Precision* s_AbsoluteTolerance, \
			int*       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int*       s_UpdateStep, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	__shared__ Precision s_RelativeError[SPB];
	__shared__ Precision s_TimeStepMultiplicator[SPB];
	
	Precision r_ErrorTolerance;
	Precision r_RelativeError;
	int       r_UpdateStep;
	
	// Relative error initialisation
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		
		if ( lsid < SPB )
			s_RelativeError[lsid] = 1e30;
	}
	__syncthreads();
	
	// Acceptance and calculation of relative error (reduction tasks)
	if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
	{
		r_RelativeError = 1e30;
		r_UpdateStep    = 1;
		
		for (int i=0; i<UD; i++)
		{
			r_ErrorTolerance = MPGOS::FMAX( s_RelativeTolerance[i]*MPGOS::FMAX( MPGOS::FABS(r_NextState[i]), MPGOS::FABS(r_ActualState[i]) ), s_AbsoluteTolerance[i] );
			r_UpdateStep     = r_UpdateStep & ( r_Error[i] < r_ErrorTolerance );
			r_RelativeError  = MPGOS::FMIN( r_RelativeError, r_ErrorTolerance / r_Error[i] );
		}
		
		atomicAnd(&(s_UpdateStep[LocalSystemID]), r_UpdateStep);
		MPGOS::atomicFMIN(&(s_RelativeError[LocalSystemID]), r_RelativeError);
	}
	__syncthreads();
	
	// New time step (with bound checks)
	Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		int lsid = threadIdx.x + j*blockDim.x;
		int gsid = lsid + blockIdx.x*SPB;
		
		if ( ( lsid < SPB ) && ( s_TerminateSystemScope[lsid] == 0 ) )
		{
			// Base time step multiplicator
			if ( s_UpdateStep[lsid] == 1 )
				s_TimeStepMultiplicator[lsid] = static_cast<Precision>(0.8) * pow(s_RelativeError[lsid], static_cast<Precision>(1.0/5.0) ); // 1.0/5.0
			else
				s_TimeStepMultiplicator[lsid] = static_cast<Precision>(0.8) * pow(s_RelativeError[lsid], static_cast<Precision>(1.0/4.0) ); // 1.0/4.0
			
			if ( isfinite(s_TimeStepMultiplicator[lsid]) == 0 )
				s_IsFinite[lsid] = 0;
			
			// Check finiteness
			if ( s_IsFinite[lsid] == 0 )
			{
				s_TimeStepMultiplicator[lsid] = SolverOptions.TimeStepShrinkLimit;
				s_UpdateStep[lsid] = 0;
				
				if ( s_TimeStep[lsid] < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
				{
					printf("Error: State is not a finite number, minimum step size reached. Try to use less stringent tolerances. (global system id: %d)\n", gsid);
					s_TerminateSystemScope[lsid] = 1;
					atomicAdd(&s_TerminatedSystemsPerBlock, 1);
				}
			} else
			{
				if ( s_TimeStep[lsid] < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
				{
					printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed! (global system id: %d, time step: %+6.5e, TSM: %+6.3e \n", gsid, SolverOptions.MinimumTimeStep, s_TimeStepMultiplicator[lsid]);
					s_UpdateStep[lsid] = 1;
				}
			}
			
			// Time step and its growth limits
			s_TimeStepMultiplicator[lsid] = MPGOS::FMIN(s_TimeStepMultiplicator[lsid], SolverOptions.TimeStepGrowLimit);
			s_TimeStepMultiplicator[lsid] = MPGOS::FMAX(s_TimeStepMultiplicator[lsid], SolverOptions.TimeStepShrinkLimit);
			
			s_NewTimeStep[lsid] = s_TimeStep[lsid] * s_TimeStepMultiplicator[lsid];
			
			s_NewTimeStep[lsid] = MPGOS::FMIN(s_NewTimeStep[lsid], SolverOptions.MaximumTimeStep);
			s_NewTimeStep[lsid] = MPGOS::FMAX(s_NewTimeStep[lsid], SolverOptions.MinimumTimeStep);
		}
	}
	__syncthreads();
}


template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_ErrorController_RK4(\
			int        s_IsFinite, \
			Precision& s_NewTimeStep, \
			Precision  InitialTimeStep, \
			int&       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int&       s_UpdateStep)
{
	const int GlobalSystemID = blockIdx.x;
	
	if ( ( threadIdx.x == 0 ) && ( s_TerminateSystemScope == 0 ) )
	{
		if ( s_IsFinite == 0 )
		{
			printf("Error: State is not a finite number. Try to use smaller step size. (global system id: %d)\n", GlobalSystemID);
			
			s_TerminateSystemScope = 1;
			atomicAdd(&s_TerminatedSystemsPerBlock, 1);
			
			s_UpdateStep = 0;
		}
		
		s_NewTimeStep = InitialTimeStep;
	}
	__syncthreads();
}


template <int UPS, int UD, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_ErrorController_RKCK45(\
			const int  LocalThreadID, \
			const int  GlobalSystemID, \
			int&       s_IsFinite, \
			Precision  s_TimeStep, \
			Precision& s_NewTimeStep, \
			Precision  r_ActualState[UD], \
			Precision  r_NextState[UD], \
			Precision  r_Error[UD], \
			Precision* s_RelativeTolerance, \
			Precision* s_AbsoluteTolerance, \
			int&       s_TerminateSystemScope, \
			int&       s_TerminatedSystemsPerBlock, \
			int&       s_UpdateStep, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	__shared__ Precision s_RelativeError;
	__shared__ Precision s_TimeStepMultiplicator;
	
	Precision r_ErrorTolerance;
	Precision r_RelativeError;
	int       r_UpdateStep;
	
	// Relative error initialisation
	if ( threadIdx.x == 0 )
		s_RelativeError = 1e30;
	__syncthreads();
	
	// Acceptance and calculation of relative error (reduction tasks)
	if ( ( LocalThreadID < UPS ) && ( s_TerminateSystemScope == 0 ) )
	{
		r_RelativeError = 1e30;
		r_UpdateStep    = 1;
		
		for (int i=0; i<UD; i++)
		{
			r_ErrorTolerance = MPGOS::FMAX( s_RelativeTolerance[i]*MPGOS::FMAX( MPGOS::FABS(r_NextState[i]), MPGOS::FABS(r_ActualState[i]) ), s_AbsoluteTolerance[i] );
			r_UpdateStep     = r_UpdateStep & ( r_Error[i] < r_ErrorTolerance );
			r_RelativeError  = MPGOS::FMIN( r_RelativeError, r_ErrorTolerance / r_Error[i] );
		}
		
		atomicAnd(&s_UpdateStep, r_UpdateStep);
		MPGOS::atomicFMIN(&s_RelativeError, r_RelativeError);
	}
	__syncthreads();
	
	if ( ( threadIdx.x == 0 ) && ( s_TerminateSystemScope == 0 ) )
	{
		// Base time step multiplicator
		if ( s_UpdateStep == 1 )
			s_TimeStepMultiplicator = static_cast<Precision>(0.8) * pow(s_RelativeError, static_cast<Precision>(1.0/5.0) ); // 1.0/5.0
		else
			s_TimeStepMultiplicator = static_cast<Precision>(0.8) * pow(s_RelativeError, static_cast<Precision>(1.0/4.0) ); // 1.0/4.0
		
		if ( isfinite(s_TimeStepMultiplicator) == 0 )
			s_IsFinite = 0;
		
		// Check finiteness
		if ( s_IsFinite == 0 )
		{
			s_TimeStepMultiplicator = SolverOptions.TimeStepShrinkLimit;
			s_UpdateStep = 0;
			
			if ( s_TimeStep < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
			{
				printf("Error: State is not a finite number, minimum step size reached. Try to use less stringent tolerances. (global system id: %d)\n", GlobalSystemID);
				s_TerminateSystemScope = 1;
				atomicAdd(&s_TerminatedSystemsPerBlock, 1);
			}
		} else
		{
			if ( s_TimeStep < (SolverOptions.MinimumTimeStep*static_cast<Precision>(1.01)) )
			{
				printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed! (global system id: %d, time step: %+6.5e, TSM: %+6.3e \n", GlobalSystemID, SolverOptions.MinimumTimeStep, s_TimeStepMultiplicator);
				s_UpdateStep = 1;
			}
		}
		
		// Time step and its growth limits
		s_TimeStepMultiplicator = MPGOS::FMIN(s_TimeStepMultiplicator, SolverOptions.TimeStepGrowLimit);
		s_TimeStepMultiplicator = MPGOS::FMAX(s_TimeStepMultiplicator, SolverOptions.TimeStepShrinkLimit);
		
		s_NewTimeStep = s_TimeStep * s_TimeStepMultiplicator;
		
		s_NewTimeStep = MPGOS::FMIN(s_NewTimeStep, SolverOptions.MaximumTimeStep);
		s_NewTimeStep = MPGOS::FMAX(s_NewTimeStep, SolverOptions.MinimumTimeStep);
	}
	__syncthreads();
}

#endif
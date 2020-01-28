#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_GLOBAL_STEPPERS_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_GLOBAL_STEPPERS_H


// ----------------------------------------------------------------------------
template <Algorithms Algorithm>
__forceinline__ __device__ void RungeKuttaStepper(IntegratorInternalVariables KernelParameters, int tid, double ActualTime, double TimeStep, double* s_SharedParameters, int* s_IntegerSharedParameters, bool& IsFinite)
{}


// ----------
template <>
__forceinline__ __device__ void RungeKuttaStepper<RK4>(IntegratorInternalVariables KernelParameters, int tid, double ActualTime, double TimeStep, double* s_SharedParameters, int* s_IntegerSharedParameters, bool& IsFinite)
{
	int TemporaryIndex1;
	double TemporaryTime;
	double HalfTimeStep = 0.5*TimeStep;
	
	
	// k1 -----
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      KernelParameters.d_NextState, KernelParameters.d_ActualState, ActualTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k2 -----
	TemporaryTime  = ActualTime + HalfTimeStep;
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + KernelParameters.d_NextState[TemporaryIndex1] * HalfTimeStep;
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      KernelParameters.d_Stages, KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k3 -----
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_NextState[TemporaryIndex1] = KernelParameters.d_NextState[TemporaryIndex1] + 2*KernelParameters.d_Stages[TemporaryIndex1];
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + KernelParameters.d_Stages[TemporaryIndex1] * HalfTimeStep;
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      KernelParameters.d_Stages, KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k4 -----
	TemporaryTime = ActualTime + TimeStep;
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{
		KernelParameters.d_NextState[TemporaryIndex1] = KernelParameters.d_NextState[TemporaryIndex1] + 2*KernelParameters.d_Stages[TemporaryIndex1];
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + KernelParameters.d_Stages[TemporaryIndex1] * TimeStep;
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      KernelParameters.d_Stages, KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// New state
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{
		KernelParameters.d_NextState[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep*d_BT_RK4[0] * ( KernelParameters.d_NextState[TemporaryIndex1] + KernelParameters.d_Stages[TemporaryIndex1] );
		
		if ( isfinite( KernelParameters.d_NextState[TemporaryIndex1] ) == 0 )
			IsFinite = 0;
		
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
}


// ----------
template <>
__forceinline__ __device__ void RungeKuttaStepper<RKCK45>(IntegratorInternalVariables KernelParameters, int tid, double ActualTime, double TimeStep, double* s_SharedParameters, int* s_IntegerSharedParameters, bool& IsFinite)
{
	int TemporaryIndex1;
	int TemporaryIndex2 = KernelParameters.SystemDimension * KernelParameters.NumberOfThreads;
	double TemporaryTime;
	
	
	// k1 -----
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      &KernelParameters.d_Stages[0], KernelParameters.d_ActualState, ActualTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k2 -----
	TemporaryTime = ActualTime + TimeStep * d_BT_RKCK45[0];
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep * ( d_BT_RKCK45[0]*KernelParameters.d_Stages[TemporaryIndex1] );
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      &KernelParameters.d_Stages[TemporaryIndex2], KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k3 -----
	TemporaryTime = ActualTime + TimeStep * d_BT_RKCK45[1];
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep * ( d_BT_RKCK45[2]*KernelParameters.d_Stages[TemporaryIndex1] + \
		                                                                                                           d_BT_RKCK45[3]*KernelParameters.d_Stages[TemporaryIndex1+TemporaryIndex2] );
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      &KernelParameters.d_Stages[2*TemporaryIndex2], KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k4 -----
	TemporaryTime = ActualTime + TimeStep * d_BT_RKCK45[4];
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep * ( d_BT_RKCK45[1]*KernelParameters.d_Stages[TemporaryIndex1] + \
		                                                                                                           d_BT_RKCK45[5]*KernelParameters.d_Stages[TemporaryIndex1+TemporaryIndex2] + \
																												   d_BT_RKCK45[6]*KernelParameters.d_Stages[TemporaryIndex1+2*TemporaryIndex2] );
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      &KernelParameters.d_Stages[3*TemporaryIndex2], KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k5 -----
	TemporaryTime = ActualTime + TimeStep;
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep * (  d_BT_RKCK45[7]*KernelParameters.d_Stages[TemporaryIndex1] + \
		                                                                                                            d_BT_RKCK45[8]*KernelParameters.d_Stages[TemporaryIndex1+TemporaryIndex2] + \
																												    d_BT_RKCK45[9]*KernelParameters.d_Stages[TemporaryIndex1+2*TemporaryIndex2] + \
																												   d_BT_RKCK45[10]*KernelParameters.d_Stages[TemporaryIndex1+3*TemporaryIndex2] );
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      &KernelParameters.d_Stages[4*TemporaryIndex2], KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// k6 -----
	TemporaryTime = ActualTime + TimeStep * d_BT_RKCK45[11];
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{	
		KernelParameters.d_State[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep * ( d_BT_RKCK45[12]*KernelParameters.d_Stages[TemporaryIndex1] + \
		                                                                                                           d_BT_RKCK45[13]*KernelParameters.d_Stages[TemporaryIndex1+TemporaryIndex2] + \
																												   d_BT_RKCK45[14]*KernelParameters.d_Stages[TemporaryIndex1+2*TemporaryIndex2] + \
																												   d_BT_RKCK45[15]*KernelParameters.d_Stages[TemporaryIndex1+3*TemporaryIndex2] + \
																												   d_BT_RKCK45[16]*KernelParameters.d_Stages[TemporaryIndex1+4*TemporaryIndex2] );
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
	
	PerThread_OdeFunction(tid, KernelParameters.NumberOfThreads, \
	                      &KernelParameters.d_Stages[5*TemporaryIndex2], KernelParameters.d_State, TemporaryTime, \
						  KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	
	
	// New state and error
	TemporaryIndex1 = tid;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{
		KernelParameters.d_NextState[TemporaryIndex1] = KernelParameters.d_ActualState[TemporaryIndex1] + TimeStep * ( KernelParameters.d_Stages[TemporaryIndex1]*d_BT_RKCK45[17] + \
		                                                                                                               KernelParameters.d_Stages[TemporaryIndex1+2*TemporaryIndex2]*d_BT_RKCK45[18] + \
																													   KernelParameters.d_Stages[TemporaryIndex1+3*TemporaryIndex2]*d_BT_RKCK45[19] + \
																													   KernelParameters.d_Stages[TemporaryIndex1+5*TemporaryIndex2]*d_BT_RKCK45[20] );
		
		KernelParameters.d_Error[TemporaryIndex1] = KernelParameters.d_Stages[TemporaryIndex1]*(d_BT_RKCK45[17]-d_BT_RKCK45[21]) + \
		                                            KernelParameters.d_Stages[TemporaryIndex1+2*TemporaryIndex2]*(d_BT_RKCK45[18]-d_BT_RKCK45[22]) + \
													KernelParameters.d_Stages[TemporaryIndex1+3*TemporaryIndex2]*(d_BT_RKCK45[19]-d_BT_RKCK45[23]) - \
													KernelParameters.d_Stages[TemporaryIndex1+4*TemporaryIndex2]*d_BT_RKCK45[24] + \
													KernelParameters.d_Stages[TemporaryIndex1+5*TemporaryIndex2]*(d_BT_RKCK45[20]-d_BT_RKCK45[25]);
		KernelParameters.d_Error[TemporaryIndex1] = TimeStep * abs( KernelParameters.d_Error[TemporaryIndex1] ) + 1e-18;
		
		if ( ( isfinite(KernelParameters.d_NextState[TemporaryIndex1]) == 0 ) || ( isfinite(KernelParameters.d_Error[TemporaryIndex1]) == 0 ) )
			IsFinite = 0;
		
		TemporaryIndex1 += KernelParameters.NumberOfThreads;
	}
}

#endif
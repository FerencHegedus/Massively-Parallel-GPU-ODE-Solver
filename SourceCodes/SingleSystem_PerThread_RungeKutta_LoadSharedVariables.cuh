#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_LOADSHAREDVARIABLES_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_LOADSHAREDVARIABLES_H


// ----------
template <AlgorithmOptions SelectedAlgorithm>
__forceinline__ __device__ void LoadSharedVariables(int* DynamicSharedMemory, IntegratorInternalVariables KernelParameters, \
	                                double*& s_SharedParameters,  int*&    s_IntegerSharedParameters, \
	                                double*& s_RelativeTolerance, double*& s_AbsoluteTolerance, \
									double*& s_EventTolerance,    int*&    s_EventDirection,    int*& s_EventStopCounter)
{}


// ----------
template <>
__forceinline__ __device__ void LoadSharedVariables<RK4>(int* DynamicSharedMemory, IntegratorInternalVariables KernelParameters, \
	                                     double*& s_SharedParameters,  int*&    s_IntegerSharedParameters, \
	                                     double*& s_RelativeTolerance, double*& s_AbsoluteTolerance, \
									     double*& s_EventTolerance,    int*&    s_EventDirection,    int*& s_EventStopCounter)
{
	s_SharedParameters = (double*)DynamicSharedMemory;
	s_EventTolerance = (double*)&s_SharedParameters[ KernelParameters.NumberOfSharedParameters ];
	s_EventDirection = (int*)&s_EventTolerance[ KernelParameters.NumberOfEvents ];
	s_EventStopCounter = (int*)&s_EventDirection[ KernelParameters.NumberOfEvents ];
	s_IntegerSharedParameters = (int*)&s_EventStopCounter[ KernelParameters.NumberOfEvents ];
	
	if (threadIdx.x==0)
	{	
		for (int i=0; i<KernelParameters.NumberOfEvents; i++)
		{
			s_EventTolerance[i]   = __ldg( &KernelParameters.d_EventTolerance[i] );
			s_EventDirection[i]   = __ldg( &KernelParameters.d_EventDirection[i] );
			s_EventStopCounter[i] = __ldg( &KernelParameters.d_EventStopCounter[i] );
		}
		
		for (int i=0; i<KernelParameters.NumberOfSharedParameters; i++)
			s_SharedParameters[i] = __ldg( &KernelParameters.d_SharedParameters[i] );
		
		for (int i=0; i<KernelParameters.NumberOfIntegerSharedParameters; i++)
			s_IntegerSharedParameters[i] = __ldg( &KernelParameters.d_IntegerSharedParameters[i] );
	}
}


// ----------
template <>
__forceinline__ __device__ void LoadSharedVariables<RKCK45>(int* DynamicSharedMemory, IntegratorInternalVariables KernelParameters, \
	                                        double*& s_SharedParameters,  int*&    s_IntegerSharedParameters, \
	                                        double*& s_RelativeTolerance, double*& s_AbsoluteTolerance, \
									        double*& s_EventTolerance,    int*&    s_EventDirection,    int*& s_EventStopCounter)
{
	s_SharedParameters = (double*)DynamicSharedMemory;
	s_RelativeTolerance = (double*)&s_SharedParameters[ KernelParameters.NumberOfSharedParameters ];
	s_AbsoluteTolerance = (double*)&s_RelativeTolerance[ KernelParameters.SystemDimension ];
	s_EventTolerance = (double*)&s_AbsoluteTolerance[ KernelParameters.SystemDimension ];
	s_EventDirection = (int*)&s_EventTolerance[ KernelParameters.NumberOfEvents ];
	s_EventStopCounter = (int*)&s_EventDirection[ KernelParameters.NumberOfEvents ];
	s_IntegerSharedParameters = (int*)&s_EventStopCounter[ KernelParameters.NumberOfEvents ];
	
	if (threadIdx.x==0)
	{
		for (int i=0; i<KernelParameters.SystemDimension; i++)
		{
			s_RelativeTolerance[i] = __ldg( &KernelParameters.d_RelativeTolerance[i] );
			s_AbsoluteTolerance[i] = __ldg( &KernelParameters.d_AbsoluteTolerance[i] );
		}
		
		for (int i=0; i<KernelParameters.NumberOfEvents; i++)
		{
			s_EventTolerance[i]   = __ldg( &KernelParameters.d_EventTolerance[i] );
			s_EventDirection[i]   = __ldg( &KernelParameters.d_EventDirection[i] );
			s_EventStopCounter[i] = __ldg( &KernelParameters.d_EventStopCounter[i] );
		}
		
		for (int i=0; i<KernelParameters.NumberOfSharedParameters; i++)
			s_SharedParameters[i] = __ldg( &KernelParameters.d_SharedParameters[i] );
		
		for (int i=0; i<KernelParameters.NumberOfIntegerSharedParameters; i++)
			s_IntegerSharedParameters[i] = __ldg( &KernelParameters.d_IntegerSharedParameters[i] );
	}
}

#endif
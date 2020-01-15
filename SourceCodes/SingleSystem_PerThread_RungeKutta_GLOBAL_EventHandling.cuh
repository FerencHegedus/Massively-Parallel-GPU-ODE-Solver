#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_EVENTHANDLING_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_EVENTHANDLING_H


// ----------------------------------------------------------------------------
template <int NE>
__forceinline__ __device__ void EventHandlingInitialisation(IntegratorInternalVariables KernelParameters, int tid)
{
	int TemporaryIndex = tid;
	for (int i=0; i<KernelParameters.NumberOfEvents; i++)
	{
		KernelParameters.d_EventCounter[TemporaryIndex] = 0;
		KernelParameters.d_EventEquilibriumCounter[TemporaryIndex] = 0;
		
		TemporaryIndex += KernelParameters.NumberOfThreads;
	}
}


// ----------
template <>
__forceinline__ __device__ void EventHandlingInitialisation<0>(IntegratorInternalVariables KernelParameters, int tid)
{}



// ----------------------------------------------------------------------------
template <int NE>
__forceinline__ __device__ void EventHandlingTimeStepControl(IntegratorInternalVariables KernelParameters, int tid, bool& UpdateRungeKuttaStep, \
                                                             double* s_EventTolerance, int* s_EventDirection, \
											                 double TimeStep, double& NewTimeStep)
{
	if ( UpdateRungeKuttaStep == 1 )
	{
		double RequiredEventHandlingTimeStep = TimeStep;
		int TemporaryIndex = tid;
		
		for (int i=0; i<KernelParameters.NumberOfEvents; i++)
		{
			if ( ( ( KernelParameters.d_ActualEventValue[TemporaryIndex] >  s_EventTolerance[i] ) && ( KernelParameters.d_NextEventValue[TemporaryIndex] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
				 ( ( KernelParameters.d_ActualEventValue[TemporaryIndex] < -s_EventTolerance[i] ) && ( KernelParameters.d_NextEventValue[TemporaryIndex] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
			{
				RequiredEventHandlingTimeStep = fmin( RequiredEventHandlingTimeStep, -KernelParameters.d_ActualEventValue[TemporaryIndex] / (KernelParameters.d_NextEventValue[TemporaryIndex]-KernelParameters.d_ActualEventValue[TemporaryIndex]) * TimeStep );
				UpdateRungeKuttaStep = 0;
				
				if ( RequiredEventHandlingTimeStep<(KernelParameters.MinimumTimeStep*1.01) )
				{
					printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, RequiredEventHandlingTimeStep, KernelParameters.MinimumTimeStep);
					UpdateRungeKuttaStep = 1;
					break;
				}
			}
			TemporaryIndex += KernelParameters.NumberOfThreads;
		}
		
		if ( UpdateRungeKuttaStep == 0 )
			NewTimeStep = RequiredEventHandlingTimeStep;
	}
}


// ----------
template <>
__forceinline__ __device__ void EventHandlingTimeStepControl<0>(IntegratorInternalVariables KernelParameters, int tid, bool& UpdateRungeKuttaStep, \
                                                                double* s_EventTolerance, int* s_EventDirection, \
													            double TimeStep, double& NewTimeStep)
{}



// ----------------------------------------------------------------------------
template <int NE>
__forceinline__ __device__ void EventHandlingUpdate(IntegratorInternalVariables KernelParameters, int tid, bool& TerminateSimulation, \
                                                    double* s_EventTolerance, int* s_EventDirection, int* s_EventStopCounter, \
									                double& ActualTime, double& TimeStep, double* s_SharedParameters, int* s_IntegerSharedParameters)
{
	int TemporaryIndex = tid;
	for (int i=0; i<KernelParameters.NumberOfEvents; i++)
	{
		if ( ( ( KernelParameters.d_ActualEventValue[TemporaryIndex] >  s_EventTolerance[i] ) && ( abs(KernelParameters.d_NextEventValue[TemporaryIndex]) < s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
		     ( ( KernelParameters.d_ActualEventValue[TemporaryIndex] < -s_EventTolerance[i] ) && ( abs(KernelParameters.d_NextEventValue[TemporaryIndex]) < s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
		{
			KernelParameters.d_EventCounter[TemporaryIndex]++;
			if ( KernelParameters.d_EventCounter[TemporaryIndex] == s_EventStopCounter[i] )
				TerminateSimulation = 1;
			
			PerThread_ActionAfterEventDetection(tid, KernelParameters.NumberOfThreads, i, KernelParameters.d_EventCounter[TemporaryIndex], \
			                                    ActualTime, TimeStep, KernelParameters.d_TimeDomain, KernelParameters.d_ActualState, \
												KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
			PerThread_EventFunction(tid, KernelParameters.NumberOfThreads, KernelParameters.d_NextEventValue, \
			                        KernelParameters.d_ActualState, ActualTime, \
									KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
		}
		
		if ( ( abs(KernelParameters.d_ActualEventValue[TemporaryIndex]) <  s_EventTolerance[i] ) && ( abs(KernelParameters.d_NextEventValue[TemporaryIndex]) > s_EventTolerance[i] ) )
			KernelParameters.d_EventEquilibriumCounter[TemporaryIndex] = 0;
		
		if ( ( abs(KernelParameters.d_ActualEventValue[TemporaryIndex]) <  s_EventTolerance[i] ) && ( abs(KernelParameters.d_NextEventValue[TemporaryIndex]) < s_EventTolerance[i] ) )
			KernelParameters.d_EventEquilibriumCounter[TemporaryIndex]++;
		
		if ( KernelParameters.d_EventEquilibriumCounter[TemporaryIndex] == KernelParameters.MaxStepInsideEvent)
			TerminateSimulation = 1;
		
		KernelParameters.d_ActualEventValue[TemporaryIndex] = KernelParameters.d_NextEventValue[TemporaryIndex];
		TemporaryIndex += KernelParameters.NumberOfThreads;
	}
}


// ----------
template <>
__forceinline__ __device__ void EventHandlingUpdate<0>(IntegratorInternalVariables KernelParameters, int tid, bool& TerminateSimulation, \
                                                       double* s_EventTolerance, int* s_EventDirection, int* s_EventStopCounter, \
										               double& ActualTime, double& TimeStep, double* s_SharedParameters, int* s_IntegerSharedParameters)
{}

#endif
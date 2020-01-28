#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_EVENTHANDLING_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_EVENTHANDLING_H


// ----------------------------------------------------------------------------
template <int NE>
__forceinline__ __device__ void EventHandlingInitialisation(int* EventCounter, int* EventEquilibriumCounter)
{
	for (int i=0; i<NE; i++)
	{
		EventCounter[i] = 0;
		EventEquilibriumCounter[i] = 0;
	}
}


// ----------
template <>
__forceinline__ __device__ void EventHandlingInitialisation<0>(int* EventCounter, int* EventEquilibriumCounter)
{}



// ----------------------------------------------------------------------------
template <int NE>
__forceinline__ __device__ void EventHandlingTimeStepControl(int tid, double* ActualEventValue, double* NextEventValue, bool& UpdateRungeKuttaStep, double* s_EventTolerance, int* s_EventDirection, double TimeStep, double& NewTimeStep, double MinimumTimeStep)
{
	if ( UpdateRungeKuttaStep == 1 )
	{
		double RequiredEventHandlingTimeStep = TimeStep;
		
		for (int i=0; i<NE; i++)
		{
			if ( ( ( ActualEventValue[i] >  s_EventTolerance[i] ) && ( NextEventValue[i] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
				 ( ( ActualEventValue[i] < -s_EventTolerance[i] ) && ( NextEventValue[i] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
			{
				RequiredEventHandlingTimeStep = fmin( RequiredEventHandlingTimeStep, -ActualEventValue[i] / (NextEventValue[i]-ActualEventValue[i]) * TimeStep );
				UpdateRungeKuttaStep = 0;
				
				if ( RequiredEventHandlingTimeStep<(MinimumTimeStep*1.01) )
				{
					printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, RequiredEventHandlingTimeStep, MinimumTimeStep);
					UpdateRungeKuttaStep = 1;
					break;
				}
			}
		}
		
		if ( UpdateRungeKuttaStep == 0 )
			NewTimeStep = RequiredEventHandlingTimeStep;
	}
}


// ----------
template <>
__forceinline__ __device__ void EventHandlingTimeStepControl<0>(int tid, double* ActualEventValue, double* NextEventValue, bool& UpdateRungeKuttaStep, double* s_EventTolerance, int* s_EventDirection, double TimeStep, double& NewTimeStep, double MinimumTimeStep)
{}



// ----------------------------------------------------------------------------
template <int NE>
__forceinline__ __device__ void EventHandlingUpdate(int tid, int NT, double* ActualEventValue, double* NextEventValue, int* EventCounter, int* EventEquilibriumCounter, bool& TerminateSimulation, int MaxStepInsideEvent, \
                                                    double* s_EventTolerance, int* s_EventDirection, int* s_EventStopCounter, \
									                double& ActualTime, double& TimeStep, double* TimeDomain, double* ActualState, double* ControlParameters, double* s_SharedParameters, int* s_IntegerSharedParameters, double* Accessories, int* IntegerAccessories)
{
	for (int i=0; i<NE; i++)
	{
		if ( ( ( ActualEventValue[i] >  s_EventTolerance[i] ) && ( abs(NextEventValue[i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
		     ( ( ActualEventValue[i] < -s_EventTolerance[i] ) && ( abs(NextEventValue[i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
		{
			EventCounter[i]++;
			if ( EventCounter[i] == s_EventStopCounter[i] )
				TerminateSimulation = 1;
			
			PerThread_ActionAfterEventDetection(tid, NT, i, EventCounter[i], ActualTime, TimeStep, TimeDomain, ActualState, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
			PerThread_EventFunction(tid, NT, NextEventValue, ActualState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
		}
		
		if ( ( abs(ActualEventValue[i]) <  s_EventTolerance[i] ) && ( abs(NextEventValue[i]) > s_EventTolerance[i] ) )
			EventEquilibriumCounter[i] = 0;
		
		if ( ( abs(ActualEventValue[i]) <  s_EventTolerance[i] ) && ( abs(NextEventValue[i]) < s_EventTolerance[i] ) )
			EventEquilibriumCounter[i]++;
		
		if ( EventEquilibriumCounter[i] == MaxStepInsideEvent)
			TerminateSimulation = 1;
		
		ActualEventValue[i] = NextEventValue[i];
	}
}


// ----------
template <>
__forceinline__ __device__ void EventHandlingUpdate<0>(int tid, int NT, double* ActualEventValue, double* NextEventValue, int* EventCounter, int* EventEquilibriumCounter, bool& TerminateSimulation, int MaxStepInsideEvent, \
                                                       double* s_EventTolerance, int* s_EventDirection, int* s_EventStopCounter, \
									                   double& ActualTime, double& TimeStep, double* TimeDomain, double* ActualState, double* ControlParameters, double* s_SharedParameters, int* s_IntegerSharedParameters, double* Accessories, int* IntegerAccessories)
{}

#endif
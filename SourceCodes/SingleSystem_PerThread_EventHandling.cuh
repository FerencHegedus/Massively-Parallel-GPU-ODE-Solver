#ifndef SINGLESYSTEM_PERTHREAD_EVENTHANDLING_H
#define SINGLESYSTEM_PERTHREAD_EVENTHANDLING_H


template <int NE, class Precision>
__forceinline__ __device__ void PerThread_EventTimeStepControl(\
			int        tid, \
			int&       r_UpdateStep, \
			int        r_TerminateSimulation, \
			Precision* r_ActualEventValue, \
			Precision* r_NextEventValue, \
			Precision* s_EventTolerance, \
			int*       s_EventDirection, \
			Precision  r_TimeStep, \
			Precision& r_NewTimeStep, \
			Precision  MinimumTimeStep)
{
	Precision EventTimeStep = r_TimeStep;
	int       IsCorrected   = 0;
	
	if ( ( r_UpdateStep == 1 ) && ( r_TerminateSimulation == 0 ) )
	{
		for (int i=0; i<NE; i++)
		{
			if ( ( ( r_ActualEventValue[i] >  s_EventTolerance[i] ) && ( r_NextEventValue[i] < -s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
				 ( ( r_ActualEventValue[i] < -s_EventTolerance[i] ) && ( r_NextEventValue[i] >  s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
			{
				EventTimeStep = MPGOS::FMIN( EventTimeStep, -r_ActualEventValue[i] / (r_NextEventValue[i]-r_ActualEventValue[i]) * r_TimeStep );
				IsCorrected   = 1;
			}
		}
	}
	
	if ( IsCorrected == 1 )
	{
		if ( EventTimeStep < MinimumTimeStep )
		{
			printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, (thread id: %d)\n", tid);
		} else
		{
			r_NewTimeStep = EventTimeStep;
			r_UpdateStep  = 0;
		}
	}
}



// ----------------------------------------------------------------------------
/*template <int NE>
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
{}*/

#endif
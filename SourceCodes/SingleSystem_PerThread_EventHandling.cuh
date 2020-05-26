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

#endif
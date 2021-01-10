#ifndef SINGLESYSTEM_PERTHREAD_EVENTHANDLING_H
#define SINGLESYSTEM_PERTHREAD_EVENTHANDLING_H


__forceinline__ __device__ void PerThread_EventTimeStepControl(int tid, \
			RegisterStruct& r, \
			SharedStruct s, \
			__MPGOS_PERTHREAD_PRECISION MinimumTimeStep)
{
	__MPGOS_PERTHREAD_PRECISION EventTimeStep = r.TimeStep;
	int       IsCorrected   = 0;

	if ( ( r.UpdateStep == 1 ) && ( r.TerminateSimulation == 0 ) )
	{
		for (int i=0; i<__MPGOS_PERTHREAD_NE; i++)
		{
			if ( ( ( r.ActualEventValue[i] >  s.EventTolerance[i] ) && ( r.NextEventValue[i] < -s.EventTolerance[i] ) && ( s.EventDirection[i] <= 0 ) ) || \
				 ( ( r.ActualEventValue[i] < -s.EventTolerance[i] ) && ( r.NextEventValue[i] >  s.EventTolerance[i] ) && ( s.EventDirection[i] >= 0 ) ) )
			{
				EventTimeStep = MPGOS::FMIN( EventTimeStep, -r.ActualEventValue[i] / (r.NextEventValue[i]-r.ActualEventValue[i]) * r.TimeStep );
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
			r.NewTimeStep = EventTimeStep;
			r.UpdateStep  = 0;
		}
	}
}

#endif

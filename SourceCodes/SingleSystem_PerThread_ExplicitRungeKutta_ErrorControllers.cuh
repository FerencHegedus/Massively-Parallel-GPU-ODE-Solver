#ifndef SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_ERRORCONTROLLERS_H
#define SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_ERRORCONTROLLERS_H

#if __MPGOS_PERTHREAD_ALGORITHM == 0
__forceinline__ __device__ void PerThread_ErrorController_RK4(int tid,RegisterStruct &r, __MPGOS_PERTHREAD_PRECISION  InitialTimeStep)
{
	if ( r.IsFinite == 0 )
	{
		printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
		r.TerminateSimulation = 1;
	}

	r.NewTimeStep = InitialTimeStep;
}
#endif

#if __MPGOS_PERTHREAD_ALGORITHM == 1
__forceinline__ __device__ void PerThread_ErrorController_RKCK45(int tid, \
		RegisterStruct &r,  \
		SharedStruct s, \
		Struct_SolverOptions SolverOptions)
{
	__MPGOS_PERTHREAD_PRECISION RelativeError = 1e30;
	__MPGOS_PERTHREAD_PRECISION ErrorTolerance;
	__MPGOS_PERTHREAD_PRECISION TimeStepMultiplicator;

	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		ErrorTolerance = MPGOS::FMAX( s.RelativeTolerance[i]*MPGOS::FMAX( MPGOS::FABS(r.NextState[i]), MPGOS::FABS(r.ActualState[i])), s.AbsoluteTolerance[i] );
		r.UpdateStep   = r.UpdateStep && ( r.Error[i] < ErrorTolerance );
		RelativeError  = MPGOS::FMIN( RelativeError, ErrorTolerance / r.Error[i] );
	}


	if ( r.UpdateStep == 1 )
		TimeStepMultiplicator = static_cast<__MPGOS_PERTHREAD_PRECISION>(0.9) * pow(RelativeError, static_cast<__MPGOS_PERTHREAD_PRECISION>(1.0/5.0) );
	else
		TimeStepMultiplicator = static_cast<__MPGOS_PERTHREAD_PRECISION>(0.9) * pow(RelativeError, static_cast<__MPGOS_PERTHREAD_PRECISION>(1.0/4.0) );

	if ( isfinite(TimeStepMultiplicator) == 0 )
		r.IsFinite = 0;


	if ( r.IsFinite == 0 )
	{
		TimeStepMultiplicator = SolverOptions.TimeStepShrinkLimit;
		r.UpdateStep = 0;

		if ( r.TimeStep < (SolverOptions.MinimumTimeStep*static_cast<__MPGOS_PERTHREAD_PRECISION>(1.01)) )
		{
			printf("Error: State is not a finite number even with the minimal step size. Try to use less stringent tolerances. (thread id: %d)\n", tid);
			r.TerminateSimulation = 1;
		}
	} else
	{
		if ( r.TimeStep < (SolverOptions.MinimumTimeStep*static_cast<__MPGOS_PERTHREAD_PRECISION>(1.01)) )
		{
			printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, r.TimeStep, SolverOptions.MinimumTimeStep);
			r.UpdateStep = 1;
		}
	}


	TimeStepMultiplicator = MPGOS::FMIN(TimeStepMultiplicator, SolverOptions.TimeStepGrowLimit);
	TimeStepMultiplicator = MPGOS::FMAX(TimeStepMultiplicator, SolverOptions.TimeStepShrinkLimit);

	r.NewTimeStep = r.TimeStep * TimeStepMultiplicator;

	r.NewTimeStep = MPGOS::FMIN(r.NewTimeStep, SolverOptions.MaximumTimeStep);
	r.NewTimeStep = MPGOS::FMAX(r.NewTimeStep, SolverOptions.MinimumTimeStep);
}
#endif

#if __MPGOS_PERTHREAD_ALGORITHM == 2
__forceinline__ __device__ void PerThread_ErrorController_DDE4(int tid,RegisterStruct &r, __MPGOS_PERTHREAD_PRECISION  InitialTimeStep)
{
	if ( r.IsFinite == 0 )
	{
		printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
		r.TerminateSimulation = 1;
	}

	r.NewTimeStep = InitialTimeStep;
}
#endif

#endif

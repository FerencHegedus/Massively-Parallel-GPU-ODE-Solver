#ifndef SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_ERRORCONTROLLERS_H
#define SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_ERRORCONTROLLERS_H


template <class Precision>
__forceinline__ __device__ void PerThread_ErrorController_RK4(\
			int        tid, \
			Precision  InitialTimeStep, \
			int&       r_IsFinite, \
			int&       r_TerminateSimulation, \
			Precision& r_NewTimeStep)
{
	if ( r_IsFinite == 0 )
	{
		printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
		r_TerminateSimulation = 1;
	}
	
	r_NewTimeStep = InitialTimeStep;
}


template <int SD, class Precision>
__forceinline__ __device__ void PerThread_ErrorController_RKCK45(\
			int        tid, \
			Precision  r_TimeStep, \
			Precision* r_ActualState, \
			Precision* r_NextState, \
			Precision* r_Error, \
			Precision* s_RelativeTolerance, \
			Precision* s_AbsoluteTolerance, \
			int&       r_UpdateStep, \
			int&       r_IsFinite, \
			int&       r_TerminateSimulation, \
			Precision& r_NewTimeStep, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	Precision RelativeError = 1e30;
	Precision ErrorTolerance;
	Precision TimeStepMultiplicator;
	
	for (int i=0; i<SD; i++)
	{
		ErrorTolerance = MPGOS::FMAX( s_RelativeTolerance[i]*MPGOS::FMAX( MPGOS::FABS(r_NextState[i]), MPGOS::FABS(r_ActualState[i])), s_AbsoluteTolerance[i] );
		r_UpdateStep   = r_UpdateStep && ( r_Error[i] < ErrorTolerance );
		RelativeError  = MPGOS::FMIN( RelativeError, ErrorTolerance / r_Error[i] );
	}
	
	
	if ( r_UpdateStep == 1 )
		TimeStepMultiplicator = static_cast<Precision>(0.9) * pow(RelativeError, static_cast<Precision>(1.0/5.0) );
	else
		TimeStepMultiplicator = static_cast<Precision>(0.9) * pow(RelativeError, static_cast<Precision>(1.0/4.0) );
	
	if ( isfinite(TimeStepMultiplicator) == 0 )
		r_IsFinite = 0;
	
	
	if ( r_IsFinite == 0 )
	{
		TimeStepMultiplicator = SolverOptions.TimeStepShrinkLimit;
		r_UpdateStep = 0;
		
		if ( r_TimeStep < (SolverOptions.MinimumTimeStep*1.01) )
		{
			printf("Error: State is not a finite number even with the minimal step size. Try to use less stringent tolerances. (thread id: %d)\n", tid);
			r_TerminateSimulation = 1;
		}
	} else
	{
		if ( r_TimeStep < (SolverOptions.MinimumTimeStep*1.01) )
		{
			printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, r_TimeStep, SolverOptions.MinimumTimeStep);
			r_UpdateStep = 1;
		}
	}
	
	
	TimeStepMultiplicator = MPGOS::FMIN(TimeStepMultiplicator, SolverOptions.TimeStepGrowLimit);
	TimeStepMultiplicator = MPGOS::FMAX(TimeStepMultiplicator, SolverOptions.TimeStepShrinkLimit);
	
	r_NewTimeStep = r_TimeStep * TimeStepMultiplicator;
	
	r_NewTimeStep = MPGOS::FMIN(r_NewTimeStep, SolverOptions.MaximumTimeStep);
	r_NewTimeStep = MPGOS::FMAX(r_NewTimeStep, SolverOptions.MinimumTimeStep);
}

#endif
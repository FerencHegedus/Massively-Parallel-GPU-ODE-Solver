#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_ERRORCONTROLLER_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_ERRORCONTROLLER_H


// ----------------------------------------------------------------------------
template <Algorithms Algorithm>
__forceinline__ __device__ void ErrorController(IntegratorInternalVariables KernelParameters, int tid, \
                                                double* s_RelativeTolerance, double* s_AbsoluteTolerance, \
								                bool& UpdateRungeKuttaStep, bool& IsFinite, bool& TerminateSimulation, \
								                double TimeStep, double& NewTimeStep)
{}


// ----------
template <>
__forceinline__ __device__ void ErrorController<RK4>(IntegratorInternalVariables KernelParameters, int tid, \
                                                     double* s_RelativeTolerance, double* s_AbsoluteTolerance, \
									                 bool& UpdateRungeKuttaStep, bool& IsFinite, bool& TerminateSimulation, \
									                 double TimeStep, double& NewTimeStep)
{
	if ( IsFinite == 0 )
	{
		printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
		TerminateSimulation = 1;
	}
	
	NewTimeStep = KernelParameters.InitialTimeStep;
}


// ----------
template <>
__forceinline__ __device__ void ErrorController<RKCK45>(IntegratorInternalVariables KernelParameters, int tid, \
                                                        double* s_RelativeTolerance, double* s_AbsoluteTolerance, \
										                bool& UpdateRungeKuttaStep, bool& IsFinite, bool& TerminateSimulation, \
										                double TimeStep, double& NewTimeStep)
{
	int TemporaryIndex = tid;
	double RelativeError = 1e30;
	double ErrorTolerance;
	double TimeStepMultiplicator;
	
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{
		ErrorTolerance = fmax( s_RelativeTolerance[i]*abs(KernelParameters.d_NextState[TemporaryIndex]), s_AbsoluteTolerance[i] );
		
		UpdateRungeKuttaStep = UpdateRungeKuttaStep && ( KernelParameters.d_Error[TemporaryIndex] < ErrorTolerance );
		RelativeError = fmin( RelativeError, ErrorTolerance / KernelParameters.d_Error[TemporaryIndex] );
		
		TemporaryIndex += KernelParameters.NumberOfThreads;
	}
	
	
	if (UpdateRungeKuttaStep == 1)
		TimeStepMultiplicator = 0.9 * pow(RelativeError, d_BT_RKCK45[0] );
	else
		TimeStepMultiplicator = 0.9 * pow(RelativeError, d_BT_RKCK45[25] );
	
	if ( isfinite(TimeStepMultiplicator) == 0 )
		IsFinite = 0;
	
	
	if ( IsFinite == 0 )
	{
		TimeStepMultiplicator = KernelParameters.TimeStepShrinkLimit;
		UpdateRungeKuttaStep = 0;
		
		if ( TimeStep<(KernelParameters.MinimumTimeStep*1.01) )
		{
			printf("Error: State is not a finite number even with the minimal step size. Try to use less stringent tolerances. (thread id: %d)\n", tid);
			TerminateSimulation = 1;
		}
	} else
	{
		if ( TimeStep<(KernelParameters.MinimumTimeStep*1.01) )
		{
			printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, TimeStep, KernelParameters.MinimumTimeStep);
			UpdateRungeKuttaStep = 1;
		}
	}
	
	
	TimeStepMultiplicator = fmin(TimeStepMultiplicator, KernelParameters.TimeStepGrowLimit);
	TimeStepMultiplicator = fmax(TimeStepMultiplicator, KernelParameters.TimeStepShrinkLimit);
	
	NewTimeStep = TimeStep * TimeStepMultiplicator;
	
	NewTimeStep = fmin(NewTimeStep, KernelParameters.MaximumTimeStep);
	NewTimeStep = fmax(NewTimeStep, KernelParameters.MinimumTimeStep);
}

#endif
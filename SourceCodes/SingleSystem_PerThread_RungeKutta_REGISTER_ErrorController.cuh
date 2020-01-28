#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_REGISTER_ERRORCONTROLLER_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_REGISTER_ERRORCONTROLLER_H


__forceinline__ __device__ void ErrorControllerRK4(int tid, double InitialTimeStep, bool& IsFinite, bool& TerminateSimulation, double& NewTimeStep)
{
	if ( IsFinite == 0 )
	{
		printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
		TerminateSimulation = 1;
	}
	
	NewTimeStep = InitialTimeStep;
}


__forceinline__ __device__ void ErrorController_RK4_PLAIN(int tid, double InitialTimeStep, bool& IsFinite, bool& TerminateSimulation, double& NewTimeStep)
{
	if ( IsFinite == 0 )
	{
		printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
		TerminateSimulation = 1;
	}
	
	NewTimeStep = InitialTimeStep;
}


#endif
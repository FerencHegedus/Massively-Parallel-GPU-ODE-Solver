#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_REGISTER_STEPPERS_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_REGISTER_STEPPERS_H


// ----------------------------------------------------------------------------
template <int NT, int SD, Algorithms Algorithm>
__forceinline__ __device__ void RungeKuttaStepperRK4(int tid, double ActualTime, double TimeStep, double* ActualState, double* NextState, double* Error, bool& IsFinite, double* ControlParameters, double* s_SharedParameters, int* s_IntegerSharedParameters, double* Accessories, int* IntegerAccessories)
{
	double X[SD];
	double k1[SD];
	
	double T;
	double dTp2 = 0.5*TimeStep;
	double dTp6 = (1.0/6.0)*TimeStep;
	
	// k1 -----
	PerThread_OdeFunction(tid, NT, NextState, ActualState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// k2 -----
	T  = ActualTime + dTp2;
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + NextState[i] * dTp2;
	
	PerThread_OdeFunction(tid, NT, k1, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// k3 -----
	#pragma unroll
	for (int i=0; i<SD; i++)
	{	
		NextState[i] = NextState[i] + 2*k1[i];
		X[i] = ActualState[i] + k1[i] * dTp2;
	}
	PerThread_OdeFunction(tid, NT, k1, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// k4 -----
	T = ActualTime + TimeStep;
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		NextState[i] = NextState[i] + 2*k1[i];
		X[i] = ActualState[i] + k1[i] * TimeStep;
	}
	PerThread_OdeFunction(tid, NT, k1, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// New state
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		NextState[i] = ActualState[i] + dTp6 * ( NextState[i] + k1[i] );
		
		if ( isfinite( NextState[i] ) == 0 )
			IsFinite = 0;
	}
}



// WILL BE DEPRECATED
template <int SD>
__forceinline__ __device__ void RungeKuttaStepper_RK4_PLAIN(double ActualTime, double TimeStep, double* ActualState, double* ControlParameters, double* NextState, bool& IsFinite)
{
	double X[SD];
	double k1[SD];
	
	double T;
	double dTp2 = 0.5*TimeStep;
	
	// k1 -----
	PerThread_OdeFunction(NextState, ActualState, ActualTime, ControlParameters);
	
	
	// k2 -----
	T  = ActualTime + dTp2;
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + NextState[i] * dTp2;
	
	PerThread_OdeFunction(k1, X, T, ControlParameters);
	
	
	// k3 -----
	#pragma unroll
	for (int i=0; i<SD; i++)
	{	
		NextState[i] = NextState[i] + 2*k1[i];
		X[i] = ActualState[i] + k1[i] * dTp2;
	}
	PerThread_OdeFunction(k1, X, T, ControlParameters);
	
	
	// k4 -----
	T = ActualTime + TimeStep;
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		NextState[i] = NextState[i] + 2*k1[i];
		X[i] = ActualState[i] + k1[i] * TimeStep;
	}
	PerThread_OdeFunction(k1, X, T, ControlParameters);
	
	
	// New state
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		NextState[i] = ActualState[i] + TimeStep*d_BT_RK4[0] * ( NextState[i] + k1[i] );
		
		if ( isfinite( NextState[i] ) == 0 )
			IsFinite = 0;
	}
}

#endif
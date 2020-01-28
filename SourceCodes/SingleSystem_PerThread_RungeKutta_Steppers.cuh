#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_STEPPERS_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_STEPPERS_H


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


// ----------------------------------------------------------------------------
template <int NT, int SD, Algorithms Algorithm>
__forceinline__ __device__ void RungeKuttaStepperRKCK45(int tid, double ActualTime, double TimeStep, double* ActualState, double* NextState, double* Error, bool& IsFinite, double* ControlParameters, double* s_SharedParameters, int* s_IntegerSharedParameters, double* Accessories, int* IntegerAccessories)
{
	double X[SD];
	double T;
	
	double k1[SD];
	double k2[SD];
	double k3[SD];
	double k4[SD];
	double k5[SD];
	double k6[SD];
	
	// k1 -----
	PerThread_OdeFunction(tid, NT, k1, ActualState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k2 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[0];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[0]*k1[i] );
	
	PerThread_OdeFunction(tid, NT, k2, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k3 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[1];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[2]*k1[i] + d_BT_RKCK45[3]*k2[i] );
	
	PerThread_OdeFunction(tid, NT, k3, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k4 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[4];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[1]*k1[i] + d_BT_RKCK45[5]*k2[i] + d_BT_RKCK45[6]*k3[i] );
	
	PerThread_OdeFunction(tid, NT, k4, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k5 -----
	T = ActualTime + TimeStep;
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[7]*k1[i] + d_BT_RKCK45[8]*k2[i] + d_BT_RKCK45[9]*k3[i] + d_BT_RKCK45[10]*k4[i] );
	
	PerThread_OdeFunction(tid, NT, k5, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k6 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[11];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[12]*k1[i] + d_BT_RKCK45[13]*k2[i] + d_BT_RKCK45[14]*k3[i] + d_BT_RKCK45[15]*k4[i] + d_BT_RKCK45[16]*k5[i] );
	
	PerThread_OdeFunction(tid, NT, k6, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// New state and error
	for (int i=0; i<SD; i++)
	{
		NextState[i] = ActualState[i] + TimeStep * ( k1[i]*d_BT_RKCK45[17] + k3[i]*d_BT_RKCK45[18] + k4[i]*d_BT_RKCK45[19] + k6[i]*d_BT_RKCK45[20] );
		
		Error[i] = k1[i]*(d_BT_RKCK45[17]-d_BT_RKCK45[21]) + k3[i]*(d_BT_RKCK45[18]-d_BT_RKCK45[22]) + k4[i]*(d_BT_RKCK45[19]-d_BT_RKCK45[23]) - k5[i]*d_BT_RKCK45[24] + k6[i]*(d_BT_RKCK45[20]-d_BT_RKCK45[25]);
		Error[i] = TimeStep * abs( Error[i] ) + 1e-18;
		
		if ( ( isfinite( NextState[i] ) == 0 ) || ( isfinite( Error[i] ) == 0 ) )
			IsFinite = 0;
	}
}


#endif
#ifndef SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_STEPPERS_H
#define SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_STEPPERS_H


// RK4 ------------------------------------------------------------------------
#if __MPGOS_PERTHREAD_ALGORITHM == 0
__forceinline__ __device__ void PerThread_Stepper_RK4(int tid, RegisterStruct &r,SharedStruct s, SharedParametersStruct SharedMemoryPointers)
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	__MPGOS_PERTHREAD_PRECISION X[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k1[__MPGOS_PERTHREAD_SD];

	__MPGOS_PERTHREAD_PRECISION T;
	__MPGOS_PERTHREAD_PRECISION dTp2 = static_cast<__MPGOS_PERTHREAD_PRECISION>(0.5)     * r.TimeStep;
	__MPGOS_PERTHREAD_PRECISION dTp6 = static_cast<__MPGOS_PERTHREAD_PRECISION>(1.0/6.0) * r.TimeStep;


	// K1 ---------------------------------------------------------------------
	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,r.NextState,r.ActualState, r.ActualTime,r,SharedMemoryPointers);

	// K2 ---------------------------------------------------------------------
	T  = r.ActualTime + dTp2;

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.NextState[i] * dTp2;

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,X,T,r,SharedMemoryPointers);

	// K3 ---------------------------------------------------------------------
	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.NextState[i] + static_cast<__MPGOS_PERTHREAD_PRECISION>(2.0)*k1[i];
		X[i] = r.ActualState[i] + k1[i] * dTp2;
	}

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,X,T,r,SharedMemoryPointers);


	// K4 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep;

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.NextState[i] + static_cast<__MPGOS_PERTHREAD_PRECISION>(2.0)*k1[i];
		X[i] = r.ActualState[i] + k1[i] * r.TimeStep;
	}

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,X,T,r,SharedMemoryPointers);
	#if __MPGOS_PERTHREAD_INTERPOLATION
		PerThread_SystemToDense(k1,r.NextDerivative,s);
	#endif


	// NEW STATE --------------------------------------------------------------
	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.ActualState[i] + dTp6 * ( r.NextState[i] + k1[i] );

		if ( isfinite( r.NextState[i] ) == 0 )
			r.IsFinite = 0;
	}
}
#endif

// RKCK 45 --------------------------------------------------------------------
#if __MPGOS_PERTHREAD_ALGORITHM == 1
__forceinline__ __device__ void PerThread_Stepper_RKCK45(int tid, RegisterStruct &r, SharedStruct s, SharedParametersStruct SharedMemoryPointers)
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	__MPGOS_PERTHREAD_PRECISION X[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION T;

	__MPGOS_PERTHREAD_PRECISION k1[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k2[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k3[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k4[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k5[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k6[__MPGOS_PERTHREAD_SD];


	// K1 ---------------------------------------------------------------------
	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,r.ActualState,r.ActualTime,r,SharedMemoryPointers);

	// K2 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep * static_cast<__MPGOS_PERTHREAD_PRECISION>(1.0/5.0);

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.TimeStep * ( static_cast<__MPGOS_PERTHREAD_PRECISION>(1.0/5.0) * k1[i] );

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k2,X,T,r,SharedMemoryPointers);

	// K3 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep * static_cast<__MPGOS_PERTHREAD_PRECISION>(3.0/10.0);

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.TimeStep * ( static_cast<__MPGOS_PERTHREAD_PRECISION>(3.0/40.0) * k1[i] + \
	                                             static_cast<__MPGOS_PERTHREAD_PRECISION>(9.0/40.0) * k2[i] );

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k3,X,T,r,SharedMemoryPointers);


	// K4 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep * static_cast<__MPGOS_PERTHREAD_PRECISION>(3.0/5.0);

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.TimeStep * ( static_cast<__MPGOS_PERTHREAD_PRECISION>(3.0/10.0)  * k1[i] + \
	                                             static_cast<__MPGOS_PERTHREAD_PRECISION>(-9.0/10.0) * k2[i] + \
												 static_cast<__MPGOS_PERTHREAD_PRECISION>(6.0/5.0)   * k3[i] );

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k4,X,T,r,SharedMemoryPointers);


	// K5 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep;

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.TimeStep * ( static_cast<__MPGOS_PERTHREAD_PRECISION>(-11.0/54.0) * k1[i] + \
	                                             static_cast<__MPGOS_PERTHREAD_PRECISION>(5.0/2.0)    * k2[i] + \
												 static_cast<__MPGOS_PERTHREAD_PRECISION>(-70.0/27.0) * k3[i] + \
												 static_cast<__MPGOS_PERTHREAD_PRECISION>(35.0/27.0)  * k4[i] );

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k5,X,T,r,SharedMemoryPointers);

	#if __MPGOS_PERTHREAD_INTERPOLATION
		PerThread_SystemToDense(k5,r.NextDerivative,s);
	#endif


	// K6 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep * static_cast<__MPGOS_PERTHREAD_PRECISION>(7.0/8.0);

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.TimeStep * ( static_cast<__MPGOS_PERTHREAD_PRECISION>(1631.0/55296.0)   * k1[i] + \
	                                             static_cast<__MPGOS_PERTHREAD_PRECISION>(175.0/512.0)      * k2[i] + \
												 static_cast<__MPGOS_PERTHREAD_PRECISION>(575.0/13824.0)    * k3[i] + \
												 static_cast<__MPGOS_PERTHREAD_PRECISION>(44275.0/110592.0) * k4[i] + \
												 static_cast<__MPGOS_PERTHREAD_PRECISION>(253.0/4096.0)     * k5[i] );

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k6,X,T,r,SharedMemoryPointers);


	// NEW STATE AND ERROR ----------------------------------------------------
	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.ActualState[i] + r.TimeStep * ( static_cast<__MPGOS_PERTHREAD_PRECISION>(37.0/378.0)   * k1[i] + \
		                                                   static_cast<__MPGOS_PERTHREAD_PRECISION>(250.0/621.0)  * k3[i] + \
														   static_cast<__MPGOS_PERTHREAD_PRECISION>(125.0/594.0)  * k4[i] + \
														   static_cast<__MPGOS_PERTHREAD_PRECISION>(512.0/1771.0) * k6[i] );

		r.Error[i] = static_cast<__MPGOS_PERTHREAD_PRECISION>(  37.0/378.0  -  2825.0/27648.0 ) * k1[i] + \
		             static_cast<__MPGOS_PERTHREAD_PRECISION>( 250.0/621.0  - 18575.0/48384.0 ) * k3[i] + \
					 static_cast<__MPGOS_PERTHREAD_PRECISION>( 125.0/594.0  - 13525.0/55296.0 ) * k4[i] + \
					 static_cast<__MPGOS_PERTHREAD_PRECISION>(   0.0        -   277.0/14336.0 ) * k5[i] + \
					 static_cast<__MPGOS_PERTHREAD_PRECISION>( 512.0/1771.0 -     1.0/4.0     ) * k6[i];
		r.Error[i] = r.TimeStep * abs( r.Error[i] ) + 1e-18;

		if ( ( isfinite( r.NextState[i] ) == 0 ) || ( isfinite( r.Error[i] ) == 0 ) )
			r.IsFinite = 0;
	}
}
#endif

// DDE4 ------------------------------------------------------------------------
#if __MPGOS_PERTHREAD_ALGORITHM == 2
__forceinline__ __device__ void PerThread_Stepper_DDE4(int tid, RegisterStruct &r, SharedStruct s, SharedParametersStruct SharedMemoryPointers)
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	__MPGOS_PERTHREAD_PRECISION X[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION k1[__MPGOS_PERTHREAD_SD];

	__MPGOS_PERTHREAD_PRECISION T;
	__MPGOS_PERTHREAD_PRECISION dTp2 = static_cast<__MPGOS_PERTHREAD_PRECISION>(0.5)     * r.TimeStep;
	__MPGOS_PERTHREAD_PRECISION dTp6 = static_cast<__MPGOS_PERTHREAD_PRECISION>(1.0/6.0) * r.TimeStep;


	// K1 ---------------------------------------------------------------------
	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,r.NextState,r.ActualState, r.ActualTime,r,SharedMemoryPointers);

	// K2 ---------------------------------------------------------------------
	T  = r.ActualTime + dTp2;

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		X[i] = r.ActualState[i] + r.NextState[i] * dTp2;

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,X,T,r,SharedMemoryPointers);

	// K3 ---------------------------------------------------------------------
	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.NextState[i] + static_cast<__MPGOS_PERTHREAD_PRECISION>(2.0)*k1[i];
		X[i] = r.ActualState[i] + k1[i] * dTp2;
	}

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,X,T,r,SharedMemoryPointers);


	// K4 ---------------------------------------------------------------------
	T = r.ActualTime + r.TimeStep;

	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.NextState[i] + static_cast<__MPGOS_PERTHREAD_PRECISION>(2.0)*k1[i];
		X[i] = r.ActualState[i] + k1[i] * r.TimeStep;
	}

	PerThread_OdeFunction(tid,__MPGOS_PERTHREAD_NT,k1,X,T,r,SharedMemoryPointers);

	//save derivatives
	PerThread_SystemToDense(k1,r.NextDerivative,s);


	// NEW STATE --------------------------------------------------------------
	#pragma unroll
	for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
	{
		r.NextState[i] = r.ActualState[i] + dTp6 * ( r.NextState[i] + k1[i] );

		if ( isfinite( r.NextState[i] ) == 0 )
			r.IsFinite = 0;
	}
}
#endif


#endif

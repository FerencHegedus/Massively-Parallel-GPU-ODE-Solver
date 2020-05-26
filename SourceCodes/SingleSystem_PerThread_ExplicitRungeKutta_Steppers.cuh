#ifndef SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_STEPPERS_H
#define SINGLESYSTEM_PERTHREAD_EXPLICITRUNGEKUTTA_STEPPERS_H


// RK4 ------------------------------------------------------------------------
template <int NT, int SD, class Precision>
__forceinline__ __device__ void PerThread_Stepper_RK4(\
			int        tid, \
			Precision  r_ActualTime, \
			Precision  r_TimeStep, \
			Precision* r_ActualState, \
			Precision* r_NextState, \
			Precision* r_Error, \
			int&       r_IsFinite, \
			Precision* r_ControlParameters, \
			Precision* gs_SharedParameters, \
			int*       gs_IntegerSharedParameters, \
			Precision* r_Accessories, \
			int*       r_IntegerAccessories)
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	Precision X[SD];
	Precision k1[SD];
	
	Precision T;
	Precision dTp2 = static_cast<Precision>(0.5)     * r_TimeStep;
	Precision dTp6 = static_cast<Precision>(1.0/6.0) * r_TimeStep;
	
	
	// K1 ---------------------------------------------------------------------
	PerThread_OdeFunction(\
		tid, \
		NT, \
		r_NextState, \
		r_ActualState, \
		r_ActualTime, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K2 ---------------------------------------------------------------------
	T  = r_ActualTime + dTp2;
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = r_ActualState[i] + r_NextState[i] * dTp2;
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k1, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K3 ---------------------------------------------------------------------
	#pragma unroll
	for (int i=0; i<SD; i++)
	{	
		r_NextState[i] = r_NextState[i] + static_cast<Precision>(2.0)*k1[i];
		X[i] = r_ActualState[i] + k1[i] * dTp2;
	}
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k1, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K4 ---------------------------------------------------------------------
	T = r_ActualTime + r_TimeStep;
	
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		r_NextState[i] = r_NextState[i] + static_cast<Precision>(2.0)*k1[i];
		X[i] = r_ActualState[i] + k1[i] * r_TimeStep;
	}
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k1, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// NEW STATE --------------------------------------------------------------
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		r_NextState[i] = r_ActualState[i] + dTp6 * ( r_NextState[i] + k1[i] );
		
		if ( isfinite( r_NextState[i] ) == 0 )
			r_IsFinite = 0;
	}
}


// RKCK 45 --------------------------------------------------------------------
template <int NT, int SD, class Precision>
__forceinline__ __device__ void PerThread_Stepper_RKCK45(\
			int        tid, \
			Precision  r_ActualTime, \
			Precision  r_TimeStep, \
			Precision* r_ActualState, \
			Precision* r_NextState, \
			Precision* r_Error, \
			int&       r_IsFinite, \
			Precision* r_ControlParameters, \
			Precision* gs_SharedParameters, \
			int*       gs_IntegerSharedParameters, \
			Precision* r_Accessories, \
			int*       r_IntegerAccessories)
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	Precision X[SD];
	Precision T;
	
	Precision k1[SD];
	Precision k2[SD];
	Precision k3[SD];
	Precision k4[SD];
	Precision k5[SD];
	Precision k6[SD];
	
	
	// K1 ---------------------------------------------------------------------
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k1, \
		r_ActualState, \
		r_ActualTime, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K2 ---------------------------------------------------------------------
	T = r_ActualTime + r_TimeStep * static_cast<Precision>(1.0/5.0);
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = r_ActualState[i] + r_TimeStep * ( static_cast<Precision>(1.0/5.0) * k1[i] );
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k2, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K3 ---------------------------------------------------------------------
	T = r_ActualTime + r_TimeStep * static_cast<Precision>(3.0/10.0);
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = r_ActualState[i] + r_TimeStep * ( static_cast<Precision>(3.0/40.0) * k1[i] + \
	                                             static_cast<Precision>(9.0/40.0) * k2[i] );
	
	PerThread_OdeFunction(tid, \
		NT, \
		k3, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K4 ---------------------------------------------------------------------
	T = r_ActualTime + r_TimeStep * static_cast<Precision>(3.0/5.0);
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = r_ActualState[i] + r_TimeStep * ( static_cast<Precision>(3.0/10.0)  * k1[i] + \
	                                             static_cast<Precision>(-9.0/10.0) * k2[i] + \
												 static_cast<Precision>(6.0/5.0)   * k3[i] );
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k4, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K5 ---------------------------------------------------------------------
	T = r_ActualTime + r_TimeStep;
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = r_ActualState[i] + r_TimeStep * ( static_cast<Precision>(-11.0/54.0) * k1[i] + \
	                                             static_cast<Precision>(5.0/2.0)    * k2[i] + \
												 static_cast<Precision>(-70.0/27.0) * k3[i] + \
												 static_cast<Precision>(35.0/27.0)  * k4[i] );
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k5, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// K6 ---------------------------------------------------------------------
	T = r_ActualTime + r_TimeStep * static_cast<Precision>(7.0/8.0);
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = r_ActualState[i] + r_TimeStep * ( static_cast<Precision>(1631.0/55296.0)   * k1[i] + \
	                                             static_cast<Precision>(175.0/512.0)      * k2[i] + \
												 static_cast<Precision>(575.0/13824.0)    * k3[i] + \
												 static_cast<Precision>(44275.0/110592.0) * k4[i] + \
												 static_cast<Precision>(253.0/4096.0)     * k5[i] );
	
	PerThread_OdeFunction(\
		tid, \
		NT, \
		k6, \
		X, \
		T, \
		r_ControlParameters, \
		gs_SharedParameters, \
		gs_IntegerSharedParameters, \
		r_Accessories, \
		r_IntegerAccessories);
	
	
	// NEW STATE AND ERROR ----------------------------------------------------
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		r_NextState[i] = r_ActualState[i] + r_TimeStep * ( static_cast<Precision>(37.0/378.0)   * k1[i] + \
		                                                   static_cast<Precision>(250.0/621.0)  * k3[i] + \
														   static_cast<Precision>(125.0/594.0)  * k4[i] + \
														   static_cast<Precision>(512.0/1771.0) * k6[i] );
		
		r_Error[i] = static_cast<Precision>(  37.0/378.0  -  2825.0/27648.0 ) * k1[i] + \
		             static_cast<Precision>( 250.0/621.0  - 18575.0/48384.0 ) * k3[i] + \
					 static_cast<Precision>( 125.0/594.0  - 13525.0/55296.0 ) * k4[i] + \
					 static_cast<Precision>(   0.0        -   277.0/14336.0 ) * k5[i] + \
					 static_cast<Precision>( 512.0/1771.0 -     1.0/4.0     ) * k6[i];
		r_Error[i] = r_TimeStep * abs( r_Error[i] ) + 1e-18;
		
		if ( ( isfinite( r_NextState[i] ) == 0 ) || ( isfinite( r_Error[i] ) == 0 ) )
			r_IsFinite = 0;
	}
}

#endif
#ifndef T1_PERTHREAD_SYSTEMDEFINITION_H
#define T1_PERTHREAD_SYSTEMDEFINITION_H

// SYSTEM
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, \
			double*    F, double*    X, double     T, \
			RegisterStruct r, double * s, int * si)
{
	F[0] = X[1];
	F[1] = X[0] - X[0]*X[0]*X[0] - r.ControlParameters[0]*X[1] + s[0]*cos(T);
}

// EVENTS
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, \
			RegisterStruct &r, double * s, int * si)
{
	r.ActualEventValue[0] = r.ActualState[1];
	r.ActualEventValue[1] = r.ActualState[0];
}

__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, \
			RegisterStruct &r, double * s, int * si)
{
	if ( r.ActualState[0] > r.Accessories[0] )
		r.Accessories[0] = r.ActualState[0];

	if ( IDX == 1 )
		r.IntegerAccessories[0]++;

	if ( (IDX ==1 ) && ( r.IntegerAccessories[0] == 2 ) )
		r.Accessories[1] = r.ActualState[1];
}

// ACCESSORIES
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
	int tid, int NT, \
	RegisterStruct &r, double * s, int * si)
{
	if ( r.ActualState[0] > r.Accessories[2] )
		r.Accessories[2] = r.ActualState[0];
}

__forceinline__ __device__ void PerThread_Initialization(\
	int tid, int NT, \
	RegisterStruct &r, double * s, int * si)
{
	r.ActualTime      = r.TimeDomain[0]; // Reset the starting point of the simulation from the lower limit of the time domain
	r.DenseOutputIndex  = 0;     // Reset the start of the filling of dense output from the beggining

	r.Accessories[0] = r.ActualState[0];
	r.Accessories[1] = r.ActualState[1];
	r.Accessories[2] = r.ActualState[0];

	r.IntegerAccessories[0] = 0; // Event counter of the second event function
}

__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, \
			RegisterStruct &r, double * s, int * si)
{

}

#endif

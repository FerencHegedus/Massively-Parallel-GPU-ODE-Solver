#ifndef T1_PERTHREAD_SYSTEMDEFINITION_H
#define T1_PERTHREAD_SYSTEMDEFINITION_H

// SYSTEM
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, \
			double*    F, double*    X, double     T, \
			RegisterStruct r, SharedParametersStruct s)
{
	F[0] = X[1];
	F[1] = X[0] - X[0]*X[0]*X[0] - r.p[0]*X[1] + s.sp[0]*cos(T);
}

// EVENTS
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, double * ef, \
			RegisterStruct &r, SharedParametersStruct s)
{
	ef[0] = r.x[1];
	ef[1] = r.x[0];
}

__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, \
			RegisterStruct &r, SharedParametersStruct s)
{
	if ( r.x[0] > r.acc[0] )
		r.acc[0] = r.x[0];

	if ( IDX == 1 )
		r.acci[0]++;

	if ( (IDX ==1 ) && ( r.acci[0] == 2 ) )
		r.acc[1] = r.x[1];
}

// ACCESSORIES
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
	int tid, int NT, \
	RegisterStruct &r, SharedParametersStruct s)
{
	if ( r.x[0] > r.acc[2] )
		r.acc[2] = r.x[0];
}

__forceinline__ __device__ void PerThread_Initialization(\
	int tid, int NT, \
	RegisterStruct &r, SharedParametersStruct s)
{
	r.t      = r.Td[0]; // Reset the starting point of the simulation from the lower limit of the time domain
	r.DenseOutputIndex  = 0;     // Reset the start of the filling of dense output from the beggining

	r.acc[0] = r.x[0];
	r.acc[1] = r.x[1];
	r.acc[2] = r.x[0];

	r.acci[0] = 0; // Event counter of the second event function
}

__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, \
			RegisterStruct &r, SharedParametersStruct s)
{

}

#endif

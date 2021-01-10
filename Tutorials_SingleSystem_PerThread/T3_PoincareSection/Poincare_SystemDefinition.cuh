#ifndef T3_PERTHREAD_SYSTEMDEFINITION_H
#define T3_PERTHREAD_SYSTEMDEFINITION_H

// SYSTEM
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, \
			double*    F, double*    X, double     T, \
			RegisterStruct r, double * s, int * si)
{
	F[0] = X[1];
	F[1] = X[0] - X[0]*X[0]*X[0] - r.ControlParameters[0]*X[1] + s[0]*cos(T);
}

__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
			int tid, int NT, RegisterStruct &r, double * s, int * si)
{

}

__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, RegisterStruct &r, double * s, int * si)
{
	r.ActualTime = r.TimeDomain[0];
}

__forceinline__ __device__ void PerThread_Finalization(\
				int tid, int NT, RegisterStruct &r, double * s, int * si)
{

}

#endif

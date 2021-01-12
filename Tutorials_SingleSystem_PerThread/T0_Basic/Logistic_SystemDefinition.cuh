#ifndef T1_PERTHREAD_SYSTEMDEFINITION_H
#define T1_PERTHREAD_SYSTEMDEFINITION_H

// SYSTEM
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, \
			float*    F, float*    X, float     T, \
			RegisterStruct r, SharedParametersStruct s)
{
	F[0] = 0.069314718*X[0]*(1.0-X[0]); //0.1*ln(2)*x[0]*(1.0-x[0])
}

// ACCESSORIES
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
	int tid, int NT, \
	RegisterStruct &r, SharedParametersStruct s)
{

}

__forceinline__ __device__ void PerThread_Initialization(\
	int tid, int NT, \
	RegisterStruct &r, SharedParametersStruct s)
{

}

__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, \
			RegisterStruct &r, SharedParametersStruct s)
{

}

#endif

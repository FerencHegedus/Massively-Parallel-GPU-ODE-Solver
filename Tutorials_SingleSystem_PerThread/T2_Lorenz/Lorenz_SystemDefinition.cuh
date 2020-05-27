#ifndef T2_PERTHREAD_SYSTEMDEFINITION_H
#define T2_PERTHREAD_SYSTEMDEFINITION_H

// SYSTEM
template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	F[0] = 10.0*( X[1]-X[0] );
	F[1] = cPAR[0]*X[0] - X[1] - X[0]*X[2];
	F[2] = X[0]*X[1] - 2.666 * X[2];
}

// EVENTS
template <class Precision>
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, Precision* EF, \
			Precision     T, Precision    dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{	
	
}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, int& UDT, \
			Precision    &T, Precision   &dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{	
	
}

// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
			int tid, int NT, int& UDT, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	
}

#endif
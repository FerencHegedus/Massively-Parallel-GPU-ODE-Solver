#ifndef REFERENCE_SYSTEMDEFINITION_H
#define REFERENCE_SYSTEMDEFINITION_H

// SYSTEM
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, double* F, double* X, double T, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	F[0] = 10.0*( X[1]-X[0] );
	F[1] = cPAR[0]*X[0] - X[1] - X[0]*X[2];
	F[2] = X[0]*X[1] - 2.666 * X[2];
}

// EVENTS
__forceinline__ __device__ void PerThread_EventFunction(int tid, int NT, double* EF, double* X, double T, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{	
	
}

__forceinline__ __device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int CNT, double &T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{	
	
}

// ACCESSORIES
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	
}

__forceinline__ __device__ void PerThread_Initialization(int tid, int NT, double& T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	
}

__forceinline__ __device__ void PerThread_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	
}

#endif
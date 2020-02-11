#ifndef PERTHREAD_SYSTEMDEFINITION_H
#define PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
__device__ void PerThread_OdeFunction(int tid, int NT, double* F, double* X, double T, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	F[0] = X[1];
	F[1] = -sPAR[0]*X[1] - (X[0]+sPAR[1]) + X[2];
	F[2] = sPAR[2]*( cPAR[0] - X[0]*sqrt(X[2]) );
}

// EVENTS
__device__ void PerThread_EventFunction(int tid, int NT, double* EF, double* X, double T, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	EF[0] = X[1]; // Poincaré section
	EF[1] = X[0]; // Impact detection
}

__device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int CNT, double &T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	if ( IDX == 1 )
		X[1] = -sPAR[3] * X[1];
}

// ACCESSORIES
__device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	if ( X[0]<ACC[0] )
		ACC[0] = X[0];
	
	if ( X[0]>ACC[1] )
		ACC[1] = X[0];
}

__device__ void PerThread_Initialization(int tid, int NT, double T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	ACC[0] = X[0];
	ACC[1] = X[0];
}

__device__ void PerThread_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	
}

#endif
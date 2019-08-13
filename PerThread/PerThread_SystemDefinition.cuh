#ifndef PERTHREAD_SYSTEMDEFINITION_H
#define PERTHREAD_SYSTEMDEFINITION_H

// SYSTEM
__device__ void PerThread_OdeFunction(int tid, int NT, double* F, double* X, double T, double* cPAR, double* sPAR, double* ACC)
{
	F(0) = X(1);
	F(1) = X(0) - X(0)*X(0)*X(0) - cPAR(0)*X(1) + sPAR(0)*cos(T);
}

// EVENTS
__device__ void PerThread_EventFunction(int tid, int NT, double* EF, double* X, double T, double* cPAR, double* sPAR, double* ACC)
{	
	EF(0) = X(1);
	EF(1) = X(0);
}

__device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int CNT, double &T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{	
	if ( X(0) > ACC(0) )
		ACC(0) = X(0);
	
	if ( (IDX==1) && (CNT==2) )
		ACC(1) = X(1);
}

// ACCESSORIES
__device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	if ( X(0) > ACC(2) )
		ACC(2) = X(0);
}

__device__ void PerThread_Initialization(int tid, int NT, double T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	ACC(0) = X(0);
	ACC(1) = X(1);
	ACC(2) = X(0);
}

__device__ void PerThread_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	
}

#endif
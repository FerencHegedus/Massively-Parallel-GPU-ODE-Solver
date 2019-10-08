#ifndef PERTHREAD_SYSTEMDEFINITION_H
#define PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
__device__ void PerThread_OdeFunction(int tid, int NT, double* F, double* X, double T, double* cPAR, double* sPAR, double* ACC)
{
	double rx1 = 1.0/X(0);
	double p   = pow(rx1, cPAR(10));
	
	double s1;
	double c1;
	sincospi(2.0*T, &s1, &c1);
	
	double s2 = sin(2.0*cPAR(11)*PI*T+cPAR(12));
	double c2 = cos(2.0*cPAR(11)*PI*T+cPAR(12));
	
	double N;
	double D;
	double rD;
	
	N = (cPAR(0)+cPAR(1)*X(1))*p - cPAR(2)*(1.0+cPAR(9)*X(1)) - cPAR(3)*rx1 - cPAR(4)*X(1)*rx1 - 1.5*(1.0-cPAR(9)*X(1)/3.0)*X(1)*X(1) - ( cPAR(5)*s1 + cPAR(6)*s2 ) * (1.0+cPAR(9)*X(1)) - X(0)*( cPAR(7)*c1 + cPAR(8)*c2 );
	D = X(0) - cPAR(9)*X(0)*X(1) + cPAR(4)*cPAR(9);
	rD = 1.0/D;
	
	
	F(0) = X(1);
	F(1) = N*rD;
}

// EVENTS
__device__ void PerThread_EventFunction(int tid, int NT, double* EF, double* X, double T, double* cPAR, double* sPAR, double* ACC)
{
	EF(0) = X(1);
}

__device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int CNT, double &T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	
}

// ACCESSORIES
__device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	if ( X(0)<ACC(2) )
	{
		ACC(2) = X(0);
		ACC(3) = T;
	}
}

__device__ void PerThread_Initialization(int tid, int NT, double T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	ACC(0) = X(0);
	ACC(1) = T;
	ACC(2) = X(0);
	ACC(3) = T;
}

__device__ void PerThread_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	TD(0) = T;
}

#endif
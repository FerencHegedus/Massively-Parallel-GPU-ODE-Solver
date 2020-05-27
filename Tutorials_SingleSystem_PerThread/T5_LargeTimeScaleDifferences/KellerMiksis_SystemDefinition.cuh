#ifndef T5_PERTHREAD_SYSTEMDEFINITION_H
#define T5_PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	Precision rx1 = 1.0/X[0];
	Precision p   = pow(rx1, cPAR[8]);
	
	Precision s1;
	Precision c1;
	sincospi(2.0*T, &s1, &c1);
	
	Precision N;
	Precision D;
	Precision rD;
	
	N = (cPAR[0]+cPAR[1]*X[1])*p - cPAR[2]*(1.0+cPAR[7]*X[1]) - cPAR[3]*rx1 - cPAR[4]*X[1]*rx1 - 1.5*(1.0-cPAR[7]*X[1]*(1.0/3.0))*X[1]*X[1] - (cPAR[5]*s1) * (1.0+cPAR[7]*X[1]) - X[0]*cPAR[6]*c1;
	D = X[0] - cPAR[7]*X[0]*X[1] + cPAR[4]*cPAR[7];
	rD = 1.0/D;
	
	F[0] = X[1];
	F[1] = N*rD;
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
	if ( X[0]>ACC[0] )
		ACC[0] = X[0];
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	T = TD[0]; // Reset the starting point of the simulation from the lower limit of the time domain
	
	ACC[0] = X[0];
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	
}

#endif
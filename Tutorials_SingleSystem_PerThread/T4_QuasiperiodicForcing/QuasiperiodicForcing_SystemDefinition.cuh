#ifndef T4_PERTHREAD_SYSTEMDEFINITION_H
#define T4_PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	double rx1 = 1.0/X[0];
	double p   = pow(rx1, cPAR[10]);
	
	double s1;
	double c1;
	sincospi(2.0*T, &s1, &c1);
	
	double s2 = sin(2.0*cPAR[11]*PI*T+cPAR[12]);
	double c2 = cos(2.0*cPAR[11]*PI*T+cPAR[12]);
	
	double N;
	double D;
	double rD;
	
	N = (cPAR[0]+cPAR[1]*X[1])*p - cPAR[2]*(1.0+cPAR[9]*X[1]) - cPAR[3]*rx1 - cPAR[4]*X[1]*rx1 - 1.5*(1.0-cPAR[9]*X[1]*(1.0/3.0))*X[1]*X[1] - ( cPAR[5]*s1 + cPAR[6]*s2 ) * (1.0+cPAR[9]*X[1]) - X[0]*( cPAR[7]*c1 + cPAR[8]*c2 );
	D = X[0] - cPAR[9]*X[0]*X[1] + cPAR[4]*cPAR[9];
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
	EF[0] = X[1];
}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, int& UDT, \
			Precision    &T, Precision   &dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	// User defined termination at local maximum
	// IDX is meaningless, as there is only 1 event function
	ACCi[0]++;
	
	if ( ACCi[0] == 1 )
		UDT = 1;
}

// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
			int tid, int NT, int& UDT, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	if ( X[0]<ACC[2] )
	{
		ACC[2] = X[0];
		ACC[3] = T;
	}
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	ACC[0] = X[0];
	ACC[1] = T;
	ACC[2] = X[0];
	ACC[3] = T;
	
	ACCi[0] = 0; // Event counter for stop condition
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	
}

#endif
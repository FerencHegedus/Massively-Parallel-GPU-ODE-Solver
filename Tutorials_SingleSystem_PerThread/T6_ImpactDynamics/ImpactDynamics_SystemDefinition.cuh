#ifndef T6_PERTHREAD_SYSTEMDEFINITION_H
#define T6_PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	F[0] = X[1];
	F[1] = -sPAR[0]*X[1] - (X[0]+sPAR[1]) + X[2];
	F[2] = sPAR[2]*( cPAR[0] - X[0]*sqrt(X[2]) );
}

// EVENTS
template <class Precision>
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, Precision* EF, \
			Precision     T, Precision    dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	EF[0] = X[1]; // Poincaré section
	EF[1] = X[0]; // Impact detection
}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, int& UDT, \
			Precision    &T, Precision   &dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	if ( IDX == 0 )
		ACCi[0]++;
	
	if ( ( ACCi[0] == 1 ) && ( IDX == 0 ) )
		UDT = 1;
	
	if ( IDX == 1 )
		X[1] = -sPAR[3] * X[1];
}

// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
			int tid, int NT, int& UDT, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	if ( X[0]<ACC[0] )
		ACC[0] = X[0];
	
	if ( X[0]>ACC[1] )
		ACC[1] = X[0];
	
	if ( abs( X[0]-ACC[2] ) < 1e-6 )
	{
		ACCi[1]++; // The new state is close to ACC[2], couner increased
	} else
	{
		ACC[2] = X[0]; // The value of X[0] is far away from that last stored value of ACC[2], replace ACC[2]
	}
	
	if ( ACCi[1] == 50 )
		UDT = 1; // Stop if X[0] is near to ACC[2] even after 50 time steps
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	T = TD[0]; // Reset the starting point of the simulation from the lower limit of the time domain
	
	ACC[0] = X[0];
	ACC[1] = X[0];
	
	ACCi[0] = 0; // Event counter of the second event function for stop condition
	
	ACC[2]  = X[0]; // Last "nearly" unmodified value
	ACCi[1] = 0;    // Equilibrium state counter
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	
}

#endif
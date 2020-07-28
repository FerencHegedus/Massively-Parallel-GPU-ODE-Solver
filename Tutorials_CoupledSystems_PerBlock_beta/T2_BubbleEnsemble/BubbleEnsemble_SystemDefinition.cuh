#ifndef PERBLOCKCOUPLING_SYSTEMDEFINITION_H
#define PERBLOCKCOUPLING_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_OdeFunction(\
			int sid, int uid, \
			Precision*    F, Precision*     X, Precision     T,             \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR, int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC, int* isACC, \
			Precision*  CPT, Precision*   CPF)
{
	Precision rx1 = 1.0/X[0];
	Precision p   = pow(rx1, uPAR[10]);
	
	Precision s1;
	Precision c1;
	sincospi(2.0*T, &s1, &c1);
	
	Precision s2 = sin(2.0*uPAR[11]*PI*T+uPAR[12]);
	Precision c2 = cos(2.0*uPAR[11]*PI*T+uPAR[12]);
	
	Precision N;
	Precision D;
	Precision rD;
	
	N = (uPAR[0]+uPAR[1]*X[1])*p - uPAR[2]*(1.0+uPAR[9]*X[1]) - uPAR[3]*rx1 - uPAR[4]*X[1]*rx1 - 1.5*(1.0-uPAR[9]*X[1]/3.0)*X[1]*X[1] - ( uPAR[5]*s1 + uPAR[6]*s2 ) * (1.0+uPAR[9]*X[1]) - X[0]*( uPAR[7]*c1 + uPAR[8]*c2 );
	D = X[0] - uPAR[9]*X[0]*X[1] + uPAR[4]*uPAR[9];
	rD = 1.0/D;
	
	F[0] = X[1];
	F[1] = N*rD;
	
	CPT[0] = -(2*X[0]*X[1]*X[1] + X[0]*X[0]*N*rD); // i=0...NC
	CPF[0] = rD;                                   // i=0...NC
}


// EVENTS
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_EventFunction(\
			int sid, int uid, Precision* EF, \
			Precision     T, Precision     dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{	
	
}

template <class Precision> // ActionAfterSuccessfulTimeStep called first!
__forceinline__ __device__ void CoupledSystems_PerBlock_ActionAfterEventDetection(\
			int sid, int uid, int IDX, int& UDT, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{	
	
}


// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_ActionAfterSuccessfulTimeStep(\
			int sid, int uid, int& UDT, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Initialization(\
			int sid, int uid, int& DOIDX, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	if ( uid == 0 )
	{
		T = TD[0]; // Reset the starting point of the simulation from the lower limit of the time domain
	}
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Finalization(\
			int sid, int uid, int& DOIDX, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	
}

#endif
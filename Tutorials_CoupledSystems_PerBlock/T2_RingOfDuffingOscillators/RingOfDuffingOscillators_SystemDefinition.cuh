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
	F[0] = X[1] - gPAR[0]*X[0];
	F[1] = -uPAR[1]*X[1] + X[0] - X[0]*X[0]*X[0] + uPAR[2]*cos(uPAR[0]*T) - gPAR[0]*X[1];
	
	// i=0...NC
	CPT[0] = X[0];
	CPF[0] = 1.0;
	
	CPT[1] = X[1];
	CPF[1] = 1.0;
}


// EVENTS
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_EventFunction(\
			int sid, int uid, Precision* EF, \
			Precision     T, Precision     dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{	
	EF[0] = X[1];
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
	if ( uid == 0 )
	{
		isACC[0]++;
		
		//if ( isACC[0] == 50 )
		//	UDT = 1;
	}
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Initialization(\
			int sid, int uid, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	if ( uid == 0 )
		isACC[0] = 0;
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Finalization(\
			int sid, int uid, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	
}

#endif
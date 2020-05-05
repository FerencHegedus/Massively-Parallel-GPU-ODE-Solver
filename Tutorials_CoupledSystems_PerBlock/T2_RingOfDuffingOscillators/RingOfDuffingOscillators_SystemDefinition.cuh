#ifndef PERBLOCKCOUPLING_SYSTEMDEFINITION_H
#define PERBLOCKCOUPLING_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_OdeFunction(\
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

// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_ActionAfterSuccessfulTimeStep(\
			Precision     T, Precision     dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Initialization(int tid, int NT, double T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	
}

#endif
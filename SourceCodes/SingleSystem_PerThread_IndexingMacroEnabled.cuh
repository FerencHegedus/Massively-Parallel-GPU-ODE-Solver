#ifndef SINGLESYSTEM_PERTHREAD_INDEXINGMACROENABLED_H
#define SINGLESYSTEM_PERTHREAD_INDEXINGMACROENABLED_H


#ifdef DATALAYOUT
	#if DATALAYOUT == GLOBAL
		#define F(i)     F[tid + i*NT]
		#define X(i)     X[tid + i*NT]
		#define cPAR(i)  cPAR[tid + i*NT]
		#define sPAR(i)  sPAR[i]
		#define sPARi(i) sPARi[i]
		#define ACC(i)   ACC[tid + i*NT]
		#define ACCi(i)  ACCi[tid + i*NT]
		#define EF(i)    EF[tid + i*NT]
		#define TD(i)    TD[tid + i*NT]
		
	#elif DATALAYOUT == REGISTER
		#define F(i)     F[i]
		#define X(i)     X[i]
		#define cPAR(i)  cPAR[i]
		#define sPAR(i)  sPAR[i]
		#define sPARi(i) sPARi[i]
		#define ACC(i)   ACC[i]
		#define ACCi(i)  ACCi[i]
		#define EF(i)    EF[i]
		#define TD(i)    TD[i]
		
	#else
		#error ERROR: Wrong definition of DATALAYOUT!
		
	#endif
#else
	#error ERROR: DATALAYOUT has not been #define-d before the indexing macro!
#endif
	
	
#endif
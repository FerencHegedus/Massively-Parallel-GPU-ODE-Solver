#ifndef SINGLESYSTEM_PERTHREAD_INDEXINGMACRODISABLED_H
#define SINGLESYSTEM_PERTHREAD_INDEXINGMACRODISABLED_H


#ifdef DATALAYOUT
	#undef F
	#undef X
	#undef cPAR
	#undef sPAR
	#undef sPARi
	#undef ACC
	#undef ACCi
	#undef EF
	#undef TD
#else
	#error ERROR: DATALAYOUT has not been #define-d before the indexing macro!
#endif


#endif
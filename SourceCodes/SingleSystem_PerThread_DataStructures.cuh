#ifndef SINGLESYSTEM_PERTHREAD_DATASTRUCTURES_H
#define SINGLESYSTEM_PERTHREAD_DATASTRUCTURES_H

enum Algorithms{ RK4, RKCK45 };

#ifndef __MPGOS_PERTHREAD_NT
	#define __MPGOS_PERTHREAD_NT 1
#endif

#ifndef __MPGOS_PERTHREAD_SD
	#define __MPGOS_PERTHREAD_SD 1
#endif

#ifndef __MPGOS_PERTHREAD_NCP
	#define __MPGOS_PERTHREAD_NCP 0
#endif

#ifndef __MPGOS_PERTHREAD_NSP
	#define __MPGOS_PERTHREAD_NSP 0
#endif

#ifndef __MPGOS_PERTHREAD_NISP
	#define __MPGOS_PERTHREAD_NISP 0
#endif

#ifndef __MPGOS_PERTHREAD_NE
	#define __MPGOS_PERTHREAD_NE 0
#endif

#ifndef __MPGOS_PERTHREAD_NA
	#define __MPGOS_PERTHREAD_NA 0
#endif

#ifndef __MPGOS_PERTHREAD_NIA
	#define __MPGOS_PERTHREAD_NIA 0
#endif

#ifndef __MPGOS_PERTHREAD_NDO
	#define __MPGOS_PERTHREAD_NDO 0
#endif

#ifndef __MPGOS_PERTHREAD_ALGORITHM
	#define __MPGOS_PERTHREAD_ALGORITHM RK4
#endif

#ifndef __MPGOS_PERTHREAD_PRECISION
	#define __MPGOS_PERTHREAD_PRECISION double
#endif

#if __MPGOS_PERTHREAD_ALGORITHM == 1
	#define __MPGOS_PERTHREAD_ADAPTIVE 1
#elif
  #define __MPGOS_PERTHREAD_ADAPTIVE 0
#endif

struct RegisterStruct
{
	__MPGOS_PERTHREAD_PRECISION TimeDomain[2];
	__MPGOS_PERTHREAD_PRECISION ActualState[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION NextState[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION Error[__MPGOS_PERTHREAD_SD];
	#if __MPGOS_PERTHREAD_NCP > 0
		__MPGOS_PERTHREAD_PRECISION ControlParameters[__MPGOS_PERTHREAD_NCP];
	#endif
	#if __MPGOS_PERTHREAD_NA > 0
		__MPGOS_PERTHREAD_PRECISION Accessories[__MPGOS_PERTHREAD_NA];
	#endif
	#if __MPGOS_PERTHREAD_NIA > 0
		__MPGOS_PERTHREAD_PRECISION IntegerAccessories[__MPGOS_PERTHREAD_NIA];
	#endif
	#if __MPGOS_PERTHREAD_NE > 0
		__MPGOS_PERTHREAD_PRECISION ActualEventValue[__MPGOS_PERTHREAD_NE];
		__MPGOS_PERTHREAD_PRECISION NextEventValue[__MPGOS_PERTHREAD_NE];
		__MPGOS_PERTHREAD_PRECISION NewTimeStepTmp;
	#endif
	#if __MPGOS_PERTHREAD_NDO > 0
		__MPGOS_PERTHREAD_PRECISION DenseOutputActualTime;
		int  DenseOutputIndex;
		int  UpdateDenseOutput;
		int  NumberOfSkippedStores;
	#endif
	__MPGOS_PERTHREAD_PRECISION ActualTime;
	__MPGOS_PERTHREAD_PRECISION TimeStep;
	__MPGOS_PERTHREAD_PRECISION NewTimeStep;
	int IsFinite;
	int TerminateSimulation;
	int UserDefinedTermination;
	int UpdateStep;
	int EndTimeDomainReached;
};

struct SharedStruct
{
	#if __MPGOS_PERTHREAD_ADAPTIVE
		__MPGOS_PERTHREAD_PRECISION RelativeTolerance[__MPGOS_PERTHREAD_SD];
		__MPGOS_PERTHREAD_PRECISION AbsoluteTolerance[__MPGOS_PERTHREAD_SD];
	#endif
	#if __MPGOS_PERTHREAD_NE > 0
		__MPGOS_PERTHREAD_PRECISION EventTolerance[__MPGOS_PERTHREAD_NE];
		int EventDirection[__MPGOS_PERTHREAD_NE];
	#endif
};


#endif
#ifndef SINGLESYSTEM_PERTHREAD_DATASTRUCTURES_H
#define SINGLESYSTEM_PERTHREAD_DATASTRUCTURES_H

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
	#define __MPGOS_PERTHREAD_ALGORITHM 0
#endif

#ifndef __MPGOS_PERTHREAD_PRECISION
	#define __MPGOS_PERTHREAD_PRECISION double
#endif

#if __MPGOS_PERTHREAD_ALGORITHM == 1
	#define __MPGOS_PERTHREAD_ADAPTIVE 1
#elif
  #define __MPGOS_PERTHREAD_ADAPTIVE 0
#endif

//-----------------------Interface with the kernel------------------------------
struct Struct_ThreadConfiguration
{
	int GridSize;
	int BlockSize;
	int NumberOfActiveThreads;
};

struct Struct_GlobalVariables
{
	__MPGOS_PERTHREAD_PRECISION* d_TimeDomain;
	__MPGOS_PERTHREAD_PRECISION* d_ActualState;
	__MPGOS_PERTHREAD_PRECISION* d_ActualTime;
	__MPGOS_PERTHREAD_PRECISION* d_ControlParameters;
	__MPGOS_PERTHREAD_PRECISION* d_SharedParameters;
	int*       d_IntegerSharedParameters;
	__MPGOS_PERTHREAD_PRECISION* d_Accessories;
	int*       d_IntegerAccessories;
	__MPGOS_PERTHREAD_PRECISION* d_RelativeTolerance;
	__MPGOS_PERTHREAD_PRECISION* d_AbsoluteTolerance;
	__MPGOS_PERTHREAD_PRECISION* d_EventTolerance;
	int*       d_EventDirection;
	int*       d_DenseOutputIndex;
	__MPGOS_PERTHREAD_PRECISION* d_DenseOutputTimeInstances;
	__MPGOS_PERTHREAD_PRECISION* d_DenseOutputStates;
};

struct Struct_SharedMemoryUsage
{
	int PreferSharedMemory;  // Default: ON
	int IsAdaptive;
};


struct Struct_SolverOptions
{
	__MPGOS_PERTHREAD_PRECISION InitialTimeStep;
	__MPGOS_PERTHREAD_PRECISION MaximumTimeStep;
	__MPGOS_PERTHREAD_PRECISION MinimumTimeStep;
	__MPGOS_PERTHREAD_PRECISION TimeStepGrowLimit;
	__MPGOS_PERTHREAD_PRECISION TimeStepShrinkLimit;
	int       DenseOutputSaveFrequency;
	__MPGOS_PERTHREAD_PRECISION DenseOutputMinimumTimeStep;
};



//-------------------------Structs for the kernel-------------------------------
struct RegisterStruct
{
	//always define
	union{
		__MPGOS_PERTHREAD_PRECISION TimeDomain[2];
		__MPGOS_PERTHREAD_PRECISION Td[2];
	}; //TimeDomain === Td

	union{
		__MPGOS_PERTHREAD_PRECISION ActualState[__MPGOS_PERTHREAD_SD];
		__MPGOS_PERTHREAD_PRECISION x[__MPGOS_PERTHREAD_SD];
	}; //ActualState === x

	__MPGOS_PERTHREAD_PRECISION NextState[__MPGOS_PERTHREAD_SD];
	__MPGOS_PERTHREAD_PRECISION Error[__MPGOS_PERTHREAD_SD];

	union{
		__MPGOS_PERTHREAD_PRECISION ActualTime;
		__MPGOS_PERTHREAD_PRECISION t;
	}; //ActualTime === t

	__MPGOS_PERTHREAD_PRECISION TimeStep;
	__MPGOS_PERTHREAD_PRECISION NewTimeStep;
	int IsFinite;
	int TerminateSimulation;
	int UserDefinedTermination;
	int UpdateStep;
	int EndTimeDomainReached;


	//if ControlParameters
	#if __MPGOS_PERTHREAD_NCP > 0
		union{
			__MPGOS_PERTHREAD_PRECISION ControlParameters[__MPGOS_PERTHREAD_NCP];
			__MPGOS_PERTHREAD_PRECISION p[__MPGOS_PERTHREAD_NCP];
		}; //ControlParameters === p
	#endif

	//if Accessories
	#if __MPGOS_PERTHREAD_NA > 0
		union
		{
			__MPGOS_PERTHREAD_PRECISION Accessories[__MPGOS_PERTHREAD_NA];
			__MPGOS_PERTHREAD_PRECISION acc[__MPGOS_PERTHREAD_NA];
		}; //Accessories === acc
	#endif

	//if integer Accessories
	#if __MPGOS_PERTHREAD_NIA > 0
		union
		{
			int IntegerAccessories[__MPGOS_PERTHREAD_NIA];
			int acci[__MPGOS_PERTHREAD_NIA];
		}; //IntegerAccessories === acci
	#endif

	//if events
	#if __MPGOS_PERTHREAD_NE > 0
		union{
			__MPGOS_PERTHREAD_PRECISION ActualEventValue[__MPGOS_PERTHREAD_NE];
			__MPGOS_PERTHREAD_PRECISION ef[__MPGOS_PERTHREAD_NE];
		};

		__MPGOS_PERTHREAD_PRECISION NextEventValue[__MPGOS_PERTHREAD_NE];
		__MPGOS_PERTHREAD_PRECISION NewTimeStepTmp;
	#endif

	//if dense output
	#if __MPGOS_PERTHREAD_NDO > 0
		__MPGOS_PERTHREAD_PRECISION DenseOutputActualTime;
		int  DenseOutputIndex;
		int  UpdateDenseOutput;
		int  NumberOfSkippedStores;
	#endif

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

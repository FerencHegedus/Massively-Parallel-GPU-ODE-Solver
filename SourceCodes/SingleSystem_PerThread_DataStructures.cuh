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

#ifndef __MPGOS_PERTHREAD_SAVEDERIVATIVES
	#define __MPGOS_PERTHREAD_SAVEDERIVATIVES 0
#endif

#ifndef __MPGOS_PERTHREAD_DOD
	#define __MPGOS_PERTHREAD_DOD __MPGOS_PERTHREAD_SD
#endif

#ifndef __MPGOS_PERTHREAD_PRECISION
	#define __MPGOS_PERTHREAD_PRECISION double
#endif

//define algorithm based on solver
#ifdef __MPGOS_PERTHREAD_SOLVER_RK4
  #ifndef __MPGOS_PERTHREAD_ALGORITHM
    #define __MPGOS_PERTHREAD_ALGORITHM 0
  #else
     #error Multiply defined solvers
  #endif
#endif

#ifdef __MPGOS_PERTHREAD_SOLVER_RKCK45
  #ifndef __MPGOS_PERTHREAD_ALGORITHM
    #define __MPGOS_PERTHREAD_ALGORITHM 1
  #else
     #error Multiply defined solvers
  #endif
#endif

#ifdef __MPGOS_PERTHREAD_SOLVER_DDE4
  #ifndef __MPGOS_PERTHREAD_ALGORITHM
    #define __MPGOS_PERTHREAD_ALGORITHM 2
  #else
     #error Multiply defined solvers
  #endif
#endif

//ALgorithm settings
#if __MPGOS_PERTHREAD_ALGORITHM == 1
	#define __MPGOS_PERTHREAD_ADAPTIVE 1
#endif
#if __MPGOS_PERTHREAD_ALGORITHM == 0 || __MPGOS_PERTHREAD_ALGORITHM == 2
	#define __MPGOS_PERTHREAD_ADAPTIVE 0
#endif

#if __MPGOS_PERTHREAD_ALGORITHM == 2
	#ifndef __MPGOS_PERTHREAD_INTERPOLATION
		#define __MPGOS_PERTHREAD_INTERPOLATION 1
	#endif
	#define __MPGOS_PERTHREAD_SAVEDERIVATIVES 1
#endif
#if __MPGOS_PERTHREAD_ALGORITHM == 0 || __MPGOS_PERTHREAD_ALGORITHM == 1
	#ifndef __MPGOS_PERTHREAD_INTERPOLATION
		#define __MPGOS_PERTHREAD_INTERPOLATION 0
	#endif
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
	__MPGOS_PERTHREAD_PRECISION* d_DenseOutputDerivatives;
	int* d_DenseToSystemIndex;
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
	__MPGOS_PERTHREAD_PRECISION DenseOutputTimeStep;
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
		__MPGOS_PERTHREAD_PRECISION ActualEventValue[__MPGOS_PERTHREAD_NE];
		__MPGOS_PERTHREAD_PRECISION NextEventValue[__MPGOS_PERTHREAD_NE];
		__MPGOS_PERTHREAD_PRECISION NewTimeStepTmp;
	#endif

	//if dense output
	#if __MPGOS_PERTHREAD_NDO > 0
		int  DenseOutputIndex;
		int  UpdateDenseOutput;
		__MPGOS_PERTHREAD_PRECISION NextDenseState[__MPGOS_PERTHREAD_DOD];
		__MPGOS_PERTHREAD_PRECISION ActualDenseState[__MPGOS_PERTHREAD_DOD];
	#endif

	//if continous output
	#if __MPGOS_PERTHREAD_INTERPOLATION
		__MPGOS_PERTHREAD_PRECISION DenseOutputActualTime;
		__MPGOS_PERTHREAD_PRECISION NextDerivative[__MPGOS_PERTHREAD_DOD];
		__MPGOS_PERTHREAD_PRECISION ActualDerivative[__MPGOS_PERTHREAD_DOD];
	#endif

	__device__ void ReadFromGlobalVariables(Struct_GlobalVariables GlobalVariables, Struct_SolverOptions SolverOptions, int tid)
	{
		//always defined
		ActualTime             = GlobalVariables.d_ActualTime[tid];
		TimeStep               = SolverOptions.InitialTimeStep;
		NewTimeStep            = SolverOptions.InitialTimeStep;
		TerminateSimulation    = 0;
		UserDefinedTermination = 0;

		#pragma unroll
		for (int i=0; i<2; i++)
			TimeDomain[i] = GlobalVariables.d_TimeDomain[tid + i*__MPGOS_PERTHREAD_NT];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
			ActualState[i] = GlobalVariables.d_ActualState[tid + i*__MPGOS_PERTHREAD_NT];

		//if control parameters
		#if __MPGOS_PERTHREAD_NCP > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NCP; i++)
				ControlParameters[i] = GlobalVariables.d_ControlParameters[tid + i*__MPGOS_PERTHREAD_NT];
		#endif

		//if accessories
		#if __MPGOS_PERTHREAD_NA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NA; i++)
				Accessories[i] = GlobalVariables.d_Accessories[tid + i*__MPGOS_PERTHREAD_NT];
		#endif

		//if integer accessories
		#if __MPGOS_PERTHREAD_NIA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NIA; i++)
				IntegerAccessories[i] = GlobalVariables.d_IntegerAccessories[tid + i*__MPGOS_PERTHREAD_NT];
		#endif

		//if dense output
		#if __MPGOS_PERTHREAD_NDO > 0
			DenseOutputIndex       = GlobalVariables.d_DenseOutputIndex[tid];

			#if __MPGOS_PERTHREAD_INTERPOLATION
				DenseOutputActualTime = TimeDomain[0];
				UpdateDenseOutput      = 0; //update at the end of step
			#else
				UpdateDenseOutput      = 1;
			#endif
		#endif
	}

	__device__ void WriteToGlobalVariables(Struct_GlobalVariables &GlobalVariables, int tid)
	{
		//always
		GlobalVariables.d_ActualTime[tid]       = ActualTime;
		#pragma unroll
		for (int i=0; i<2; i++)
			GlobalVariables.d_TimeDomain[tid + i*__MPGOS_PERTHREAD_NT] = TimeDomain[i];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
			GlobalVariables.d_ActualState[tid + i*__MPGOS_PERTHREAD_NT] = ActualState[i];

		//if control parameters
		#if __MPGOS_PERTHREAD_NCP > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NCP; i++)
				GlobalVariables.d_ControlParameters[tid + i*__MPGOS_PERTHREAD_NT] = ControlParameters[i];
		#endif

		//if accessories
		#if __MPGOS_PERTHREAD_NA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NA; i++)
				GlobalVariables.d_Accessories[tid + i*__MPGOS_PERTHREAD_NT] = Accessories[i];
		#endif

		//if integere accessories
		#if __MPGOS_PERTHREAD_NIA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NIA; i++)
				GlobalVariables.d_IntegerAccessories[tid + i*__MPGOS_PERTHREAD_NT] = IntegerAccessories[i];
		#endif

		#if __MPGOS_PERTHREAD_NDO > 0
			GlobalVariables.d_DenseOutputIndex[tid] = DenseOutputIndex;
		#endif
	}
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

	#if __MPGOS_PERTHREAD_NDO > 0
		int DenseToSystemIndex[__MPGOS_PERTHREAD_DOD];
	#endif
};

struct SharedParametersStruct
{
	__MPGOS_PERTHREAD_PRECISION *sp;
	int* spi ;

	__device__ void ReadFromGlobalVariables(Struct_GlobalVariables GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage)
	{
		extern __shared__ int DynamicSharedMemory[];
		int MemoryShift;

		sp = (__MPGOS_PERTHREAD_PRECISION*)&DynamicSharedMemory;
		MemoryShift = (SharedMemoryUsage.PreferSharedMemory  == 1 ? __MPGOS_PERTHREAD_NSP : 0);

		spi = (int*)&sp[MemoryShift];



		// Initialise shared parameters
		if ( SharedMemoryUsage.PreferSharedMemory == 0 )
		{
			sp        = GlobalVariables.d_SharedParameters;
			spi = GlobalVariables.d_IntegerSharedParameters;
		} else
		{
			const int MaxElementNumber = max( __MPGOS_PERTHREAD_NSP, __MPGOS_PERTHREAD_NISP );
			const int LaunchesSP       = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);

			#pragma unroll
			for (int i=0; i<LaunchesSP; i++)
			{
				int ltid = threadIdx.x + i*blockDim.x;

				if ( ltid < __MPGOS_PERTHREAD_NSP )
					sp[ltid] = GlobalVariables.d_SharedParameters[ltid];

				if ( ltid < __MPGOS_PERTHREAD_NISP )
					spi[ltid] = GlobalVariables.d_IntegerSharedParameters[ltid];
			}
		}
	}
};


#endif

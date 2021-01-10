#ifndef SINGLESYSTEM_PERTHREAD_SOLVER_H
#define SINGLESYSTEM_PERTHREAD_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "SingleSystem_PerThread_DenseOutput.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_Steppers.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_ErrorControllers.cuh"
#include "SingleSystem_PerThread_EventHandling.cuh"

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
	__MPGOS_PERTHREAD_PRECISION* SharedParameters;
	int* IntegerSharedParameters;
	#if __MPGOS_PERTHREAD_ADAPTIVE
		__shared__ __MPGOS_PERTHREAD_PRECISION RelativeTolerance[__MPGOS_PERTHREAD_SD];
		__shared__ __MPGOS_PERTHREAD_PRECISION AbsoluteTolerance[__MPGOS_PERTHREAD_SD];
	#endif
	#if __MPGOS_PERTHREAD_NE > 0
		__shared__ __MPGOS_PERTHREAD_PRECISION EventTolerance[__MPGOS_PERTHREAD_NE];
		__shared__ int EventDirection[__MPGOS_PERTHREAD_NE];
	#endif
};

__global__ void SingleSystem_PerThread(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int tid = threadIdx.x + blockIdx.x*blockDim.x;


	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    MINIMUM ALLOCABLE MEMORY IS 1
	SharedStruct s;

	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;

	s.SharedParameters = (Precision*)&DynamicSharedMemory;
	MemoryShift = (SharedMemoryUsage.PreferSharedMemory  == 1 ? __MPGOS_PERTHREAD_NSP : 0);

	s.IntegerSharedParameters = (int*)&gs_SharedParameters[MemoryShift];

	// Initialise tolerances of adaptive solvers
	#if __MPGOS_PERTHREAD_ADAPTIVE
		const int LaunchesSD = __MPGOS_PERTHREAD_SD / blockDim.x + (__MPGOS_PERTHREAD_SD % blockDim.x == 0 ? 0 : 1);
		#pragma unroll
		for (int j=0; j<LaunchesSD; j++)
		{
			int ltid = threadIdx.x + j*blockDim.x;

			if ( ltid < __MPGOS_PERTHREAD_SD)
			{
				s.RelativeTolerance[ltid] = GlobalVariables.d_RelativeTolerance[ltid];
				s.AbsoluteTolerance[ltid] = GlobalVariables.d_AbsoluteTolerance[ltid];
			}
		}
	#endif

	// Initialise shared event handling variables
	#if __MPGOS_PERTHREAD_NE > 0
		const int LaunchesNE = __MPGOS_PERTHREAD_NE / blockDim.x + (__MPGOS_PERTHREAD_NE % blockDim.x == 0 ? 0 : 1);
		#pragma unroll
		for (int j=0; j<LaunchesNE; j++)
		{
			int ltid = threadIdx.x + j*blockDim.x;

			if ( ltid < __MPGOS_PERTHREAD_SD)
			{
				s.EventTolerance[ltid] = GlobalVariables.d_EventTolerance[ltid];
				s.EventDirection[ltid] = GlobalVariables.d_EventDirection[ltid];
			}
		}
	#endif

	// Initialise shared parameters
	if ( SharedMemoryUsage.PreferSharedMemory == 0 )
	{
		s.SharedParameters        = GlobalVariables.d_SharedParameters;
		s.IntegerSharedParameters = GlobalVariables.d_IntegerSharedParameters;
	} else
	{
		const int MaxElementNumber = max( __MPGOS_PERTHREAD_NSP, __MPGOS_PERTHREAD_NISP );
		const int LaunchesSP       = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);

		#pragma unroll
		for (int i=0; i<LaunchesSP; i++)
		{
			int ltid = threadIdx.x + i*blockDim.x;

			if ( ltid < __MPGOS_PERTHREAD_NSP )
				s.SharedParameters[ltid] = GlobalVariables.d_SharedParameters[ltid];

			if ( ltid < __MPGOS_PERTHREAD_NISP )
				s.IntegerSharedParameters[ltid] = GlobalVariables.d_IntegerSharedParameters[ltid];
		}
	}

	if (tid < ThreadConfiguration.NumberOfActiveThreads)
	{
		// REGISTER MEMORY MANAGEMENT -----------------------------------------
		//    MINIMUM ALLOCABLE MEMORY IS 1
		RegisterStruct r;


		#pragma unroll
		for (int i=0; i<2; i++)
			r_TimeDomain[i] = GlobalVariables.d_TimeDomain[tid + i*NT];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
			r_ActualState[i] = GlobalVariables.d_ActualState[tid + i*NT];

		#if __MPGOS_PERTHREAD_NCP > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NCP; i++)
				r_ControlParameters[i] = GlobalVariables.d_ControlParameters[tid + i*NT];
		#endif

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_NA; i++)
			r_Accessories[i] = GlobalVariables.d_Accessories[tid + i*NT];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_NIA; i++)
			r_IntegerAccessories[i] = GlobalVariables.d_IntegerAccessories[tid + i*NT];

		r_ActualTime             = GlobalVariables.d_ActualTime[tid];
		r_TimeStep               = SolverOptions.InitialTimeStep;
		r_NewTimeStep            = SolverOptions.InitialTimeStep;
		r_DenseOutputIndex       = GlobalVariables.d_DenseOutputIndex[tid];
		r_UpdateDenseOutput      = 1;
		r_NumberOfSkippedStores  = 0;
		r_TerminateSimulation    = 0;
		r_UserDefinedTermination = 0;


		// INITIALISATION -----------------------------------------------------
		PerThread_Initialization<Precision>(\
			tid, \
			NT, \
			r_DenseOutputIndex, \
			r_ActualTime, \
			r_TimeStep, \
			r_TimeDomain, \
			r_ActualState, \
			r_ControlParameters, \
			gs_SharedParameters, \
			gs_IntegerSharedParameters, \
			r_Accessories, \
			r_IntegerAccessories);

		if ( __MPGOS_PERTHREAD_NE > 0 )
		{
			PerThread_EventFunction<Precision>(\
				tid, \
				NT, \
				r_ActualEventValue, \
				r_ActualTime, \
				r_TimeStep, \
				r_TimeDomain, \
				r_ActualState, \
				r_ControlParameters, \
				gs_SharedParameters, \
				gs_IntegerSharedParameters, \
				r_Accessories, \
				r_IntegerAccessories);
		}

		if ( NDO > 0 )
		{
			PerThread_StoreDenseOutput<NT, __MPGOS_PERTHREAD_SD, NDO, Precision>(\
				tid, \
				r_UpdateDenseOutput, \
				r_DenseOutputIndex, \
				GlobalVariables.d_DenseOutputTimeInstances, \
				r_ActualTime, \
				GlobalVariables.d_DenseOutputStates, \
				r_ActualState, \
				r_NumberOfSkippedStores, \
				r_DenseOutputActualTime, \
				SolverOptions.DenseOutputMinimumTimeStep, \
				r_TimeDomain[1]);
		}


		// SOLVER MANAGEMENT --------------------------------------------------
		while ( r_TerminateSimulation == 0 )
		{
			// INITIALISE TIME STEPPING ---------------------------------------
			r_UpdateStep           = 1;
			r_IsFinite             = 1;
			r_EndTimeDomainReached = 0;

			r_TimeStep = r_NewTimeStep;

			if ( r_TimeStep > ( r_TimeDomain[1] - r_ActualTime ) )
			{
				r_TimeStep = r_TimeDomain[1] - r_ActualTime;
				r_EndTimeDomainReached = 1;
			}


			// STEPPER --------------------------------------------------------
			if ( Algorithm == RK4 )
			{
				PerThread_Stepper_RK4<NT,__MPGOS_PERTHREAD_SD,Precision>(\
					tid, \
					r_ActualTime, \
					r_TimeStep, \
					r_ActualState, \
					r_NextState, \
					r_Error, \
					r_IsFinite, \
					r_ControlParameters, \
					gs_SharedParameters, \
					gs_IntegerSharedParameters, \
					r_Accessories, \
					r_IntegerAccessories);

				PerThread_ErrorController_RK4<Precision>(\
					tid, \
					SolverOptions.InitialTimeStep, \
					r_IsFinite, \
					r_TerminateSimulation, \
					r_NewTimeStep);
			}

			if ( Algorithm == RKCK45 )
			{
				PerThread_Stepper_RKCK45<NT,__MPGOS_PERTHREAD_SD,Precision>(\
					tid, \
					r_ActualTime, \
					r_TimeStep, \
					r_ActualState, \
					r_NextState, \
					r_Error, \
					r_IsFinite, \
					r_ControlParameters, \
					gs_SharedParameters, \
					gs_IntegerSharedParameters, \
					r_Accessories, \
					r_IntegerAccessories);

				PerThread_ErrorController_RKCK45<__MPGOS_PERTHREAD_SD,Precision>(\
					tid, \
					r_TimeStep, \
					r_ActualState, \
					r_NextState, \
					r_Error, \
					s_RelativeTolerance, \
					s_AbsoluteTolerance, \
					r_UpdateStep, \
					r_IsFinite, \
					r_TerminateSimulation, \
					r_NewTimeStep, \
					SolverOptions);
			}


			// NEW EVENT VALUE AND TIME STEP CONTROL---------------------------
			if ( __MPGOS_PERTHREAD_NE > 0 )
			{
				PerThread_EventFunction<Precision>(\
					tid, \
					NT, \
					r_NextEventValue, \
					r_ActualTime+r_TimeStep, \
					r_TimeStep, \
					r_TimeDomain, \
					r_NextState, \
					r_ControlParameters, \
					gs_SharedParameters, \
					gs_IntegerSharedParameters, \
					r_Accessories, \
					r_IntegerAccessories);

				PerThread_EventTimeStepControl<__MPGOS_PERTHREAD_NE,Precision>(\
					tid, \
					r_UpdateStep, \
					r_TerminateSimulation, \
					r_ActualEventValue, \
					r_NextEventValue, \
					s_EventTolerance, \
					s_EventDirection, \
					r_TimeStep, \
					r_NewTimeStep, \
					SolverOptions.MinimumTimeStep);
			}


			// UPDATE PROCESS -------------------------------------------------
			if ( r_UpdateStep == 1 )
			{
				r_ActualTime += r_TimeStep;

				for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
					r_ActualState[i] = r_NextState[i];

				PerThread_ActionAfterSuccessfulTimeStep<Precision>(\
					tid, \
					NT, \
					r_UserDefinedTermination, \
					r_ActualTime, \
					r_TimeStep, \
					r_TimeDomain, \
					r_ActualState, \
					r_ControlParameters, \
					gs_SharedParameters, \
					gs_IntegerSharedParameters, \
					r_Accessories, \
					r_IntegerAccessories);

				if ( __MPGOS_PERTHREAD_NE > 0 )
				{
					for (int i=0; i<__MPGOS_PERTHREAD_NE; i++)
					{
						if ( ( ( r_ActualEventValue[i] >  s_EventTolerance[i] ) && ( abs(r_NextEventValue[i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
							 ( ( r_ActualEventValue[i] < -s_EventTolerance[i] ) && ( abs(r_NextEventValue[i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
						{
							PerThread_ActionAfterEventDetection<Precision>(\
								tid, \
								NT, \
								i, \
								r_UserDefinedTermination, \
								r_ActualTime, \
								r_TimeStep, \
								r_TimeDomain, \
								r_ActualState, \
								r_ControlParameters, \
								gs_SharedParameters, \
								gs_IntegerSharedParameters, \
								r_Accessories, \
								r_IntegerAccessories);
						}
					}

					PerThread_EventFunction<Precision>(\
						tid, \
						NT, \
						r_NextEventValue, \
						r_ActualTime, \
						r_TimeStep, \
						r_TimeDomain, \
						r_ActualState, \
						r_ControlParameters, \
						gs_SharedParameters, \
						gs_IntegerSharedParameters, \
						r_Accessories,
						r_IntegerAccessories);

					for (int i=0; i<__MPGOS_PERTHREAD_NE; i++)
						r_ActualEventValue[i] = r_NextEventValue[i];
				}

				if ( NDO > 0 )
				{
					PerThread_DenseOutputStorageCondition<NDO, Precision>(\
						r_ActualTime, \
						r_DenseOutputActualTime, \
						r_DenseOutputIndex, \
						r_NumberOfSkippedStores, \
						r_EndTimeDomainReached, \
						r_UserDefinedTermination, \
						r_UpdateDenseOutput, \
						SolverOptions);

					PerThread_StoreDenseOutput<NT, __MPGOS_PERTHREAD_SD, NDO, Precision>(\
						tid, \
						r_UpdateDenseOutput, \
						r_DenseOutputIndex, \
						GlobalVariables.d_DenseOutputTimeInstances, \
						r_ActualTime, \
						GlobalVariables.d_DenseOutputStates, \
						r_ActualState, \
						r_NumberOfSkippedStores, \
						r_DenseOutputActualTime, \
						SolverOptions.DenseOutputMinimumTimeStep, \
						r_TimeDomain[1]);
				}

				if ( ( r_EndTimeDomainReached == 1 ) || ( r_UserDefinedTermination == 1 ) )
					r_TerminateSimulation = 1;
			}
		}


		// FINALISATION -----------------------------------------------------------
		PerThread_Finalization(\
			tid, \
			NT, \
			r_DenseOutputIndex, \
			r_ActualTime, \
			r_TimeStep, \
			r_TimeDomain, \
			r_ActualState, \
			r_ControlParameters, \
			gs_SharedParameters, \
			gs_IntegerSharedParameters, \
			r_Accessories, \
			r_IntegerAccessories);


		// WRITE DATA BACK TO GLOBAL MEMORY ---------------------------------------
		#pragma unroll
		for (int i=0; i<2; i++)
			GlobalVariables.d_TimeDomain[tid + i*NT] = r_TimeDomain[i];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
			GlobalVariables.d_ActualState[tid + i*NT] = r_ActualState[i];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_NCP; i++)
			GlobalVariables.d_ControlParameters[tid + i*NT] = r_ControlParameters[i];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_NA; i++)
			GlobalVariables.d_Accessories[tid + i*NT] = r_Accessories[i];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_NIA; i++)
			GlobalVariables.d_IntegerAccessories[tid + i*NT] = r_IntegerAccessories[i];

		GlobalVariables.d_ActualTime[tid]       = r_ActualTime;
		GlobalVariables.d_DenseOutputIndex[tid] = r_DenseOutputIndex;
	}
}

#endif

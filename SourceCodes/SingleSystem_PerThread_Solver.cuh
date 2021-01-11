#ifndef SINGLESYSTEM_PERTHREAD_SOLVER_H
#define SINGLESYSTEM_PERTHREAD_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "SingleSystem_PerThread_DenseOutput.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_Steppers.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_ErrorControllers.cuh"
#include "SingleSystem_PerThread_EventHandling.cuh"

__global__ void SingleSystem_PerThread(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int tid = threadIdx.x + blockIdx.x*blockDim.x;


	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    MINIMUM ALLOCABLE MEMORY IS 1
	__shared__ SharedStruct s;
	__MPGOS_PERTHREAD_PRECISION* SharedParameters;
	int* IntegerSharedParameters;

	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;

	SharedParameters = (__MPGOS_PERTHREAD_PRECISION*)&DynamicSharedMemory;
	MemoryShift = (SharedMemoryUsage.PreferSharedMemory  == 1 ? __MPGOS_PERTHREAD_NSP : 0);

	IntegerSharedParameters = (int*)&SharedParameters[MemoryShift];

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

	// Initialise shared event handling variables if event handling is necessary
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
		SharedParameters        = GlobalVariables.d_SharedParameters;
		IntegerSharedParameters = GlobalVariables.d_IntegerSharedParameters;
	} else
	{
		const int MaxElementNumber = max( __MPGOS_PERTHREAD_NSP, __MPGOS_PERTHREAD_NISP );
		const int LaunchesSP       = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);

		#pragma unroll
		for (int i=0; i<LaunchesSP; i++)
		{
			int ltid = threadIdx.x + i*blockDim.x;

			if ( ltid < __MPGOS_PERTHREAD_NSP )
				SharedParameters[ltid] = GlobalVariables.d_SharedParameters[ltid];

			if ( ltid < __MPGOS_PERTHREAD_NISP )
				IntegerSharedParameters[ltid] = GlobalVariables.d_IntegerSharedParameters[ltid];
		}
	}

	if (tid < ThreadConfiguration.NumberOfActiveThreads)
	{
		// REGISTER MEMORY MANAGEMENT -----------------------------------------
		RegisterStruct r;

		//always defined
		r.ActualTime             = GlobalVariables.d_ActualTime[tid];
		r.TimeStep               = SolverOptions.InitialTimeStep;
		r.NewTimeStep            = SolverOptions.InitialTimeStep;
		r.TerminateSimulation    = 0;
		r.UserDefinedTermination = 0;

		#pragma unroll
		for (int i=0; i<2; i++)
			r.TimeDomain[i] = GlobalVariables.d_TimeDomain[tid + i*__MPGOS_PERTHREAD_NT];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
			r.ActualState[i] = GlobalVariables.d_ActualState[tid + i*__MPGOS_PERTHREAD_NT];

		//if control parameters
		#if __MPGOS_PERTHREAD_NCP > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NCP; i++)
				r.ControlParameters[i] = GlobalVariables.d_ControlParameters[tid + i*__MPGOS_PERTHREAD_NT];
		#endif

		//if accessories
		#if __MPGOS_PERTHREAD_NA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NA; i++)
				r.Accessories[i] = GlobalVariables.d_Accessories[tid + i*__MPGOS_PERTHREAD_NT];
		#endif

		//if integer accessories
		#if __MPGOS_PERTHREAD_NIA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NIA; i++)
				r.IntegerAccessories[i] = GlobalVariables.d_IntegerAccessories[tid + i*__MPGOS_PERTHREAD_NT];
		#endif

		//if dense output
		#if __MPGOS_PERTHREAD_NDO > 0
			r.DenseOutputIndex       = GlobalVariables.d_DenseOutputIndex[tid];
			r.UpdateDenseOutput      = 1;
			r.NumberOfSkippedStores  = 0;
		#endif


		// INITIALISATION -----------------------------------------------------
		PerThread_Initialization(tid,__MPGOS_PERTHREAD_NT,r,SharedParameters,IntegerSharedParameters);

		#if __MPGOS_PERTHREAD_NE > 0
			PerThread_EventFunction(tid,__MPGOS_PERTHREAD_NT,r.ActualEventValue,r,SharedParameters,IntegerSharedParameters);
		#endif

		#if __MPGOS_PERTHREAD_NDO > 0
			PerThread_StoreDenseOutput(tid,r, \
				GlobalVariables.d_DenseOutputTimeInstances, \
				GlobalVariables.d_DenseOutputStates, \
				SolverOptions.DenseOutputMinimumTimeStep);
		#endif


		// SOLVER MANAGEMENT --------------------------------------------------
		while ( r.TerminateSimulation == 0 )
		{
			// INITIALISE TIME STEPPING ---------------------------------------
			r.UpdateStep           = 1;
			r.IsFinite             = 1;
			r.EndTimeDomainReached = 0;

			r.TimeStep = r.NewTimeStep;

			if ( r.TimeStep > ( r.TimeDomain[1] - r.ActualTime ) )
			{
				r.TimeStep = r.TimeDomain[1] - r.ActualTime;
				r.EndTimeDomainReached = 1;
			}


			// STEPPER --------------------------------------------------------
			#if __MPGOS_PERTHREAD_ALGORITHM == 0
				PerThread_Stepper_RK4(tid,r,SharedParameters,IntegerSharedParameters);
				PerThread_ErrorController_RK4(tid,r,SolverOptions.InitialTimeStep);
			#endif

			#if __MPGOS_PERTHREAD_ALGORITHM == 1
				PerThread_Stepper_RKCK45(tid,r,SharedParameters,IntegerSharedParameters);
				PerThread_ErrorController_RKCK45(tid,r,s,SolverOptions);
			#endif


			// NEW EVENT VALUE AND TIME STEP CONTROL---------------------------
			#if __MPGOS_PERTHREAD_NE > 0
				r.NewTimeStepTmp = r.ActualTime+r.TimeStep;
				PerThread_EventFunction(tid,__MPGOS_PERTHREAD_NT,r.ActualEventValue,r,SharedParameters,IntegerSharedParameters);
				PerThread_EventTimeStepControl(tid,r,s,SolverOptions.MinimumTimeStep);
			#endif


			// UPDATE PROCESS -------------------------------------------------
			if ( r.UpdateStep == 1 )
			{
				r.ActualTime += r.TimeStep;

				for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
					r.ActualState[i] = r.NextState[i];

				PerThread_ActionAfterSuccessfulTimeStep(tid,__MPGOS_PERTHREAD_NT,r,SharedParameters,IntegerSharedParameters);

				#if __MPGOS_PERTHREAD_NE > 0
					for (int i=0; i<__MPGOS_PERTHREAD_NE; i++)
					{
						if ( ( ( r.ActualEventValue[i] >  s.EventTolerance[i] ) && ( abs(r.NextEventValue[i]) < s.EventTolerance[i] ) && ( s.EventDirection[i] <= 0 ) ) || \
							 ( ( r.ActualEventValue[i] < -s.EventTolerance[i] ) && ( abs(r.NextEventValue[i]) < s.EventTolerance[i] ) && ( s.EventDirection[i] >= 0 ) ) )
						{
							PerThread_ActionAfterEventDetection(tid,__MPGOS_PERTHREAD_NT,i,r,SharedParameters,IntegerSharedParameters);
						}
					}

					PerThread_EventFunction(tid,__MPGOS_PERTHREAD_NT,r.ActualEventValue,r,SharedParameters,IntegerSharedParameters);

					for (int i=0; i<__MPGOS_PERTHREAD_NE; i++)
						r.ActualEventValue[i] = r.NextEventValue[i];
				#endif

				#if __MPGOS_PERTHREAD_NDO > 0
					PerThread_DenseOutputStorageCondition(r,SolverOptions);
					PerThread_StoreDenseOutput(tid,r, \
						GlobalVariables.d_DenseOutputTimeInstances, \
						GlobalVariables.d_DenseOutputStates, \
						SolverOptions.DenseOutputMinimumTimeStep);
				#endif

				if ( ( r.EndTimeDomainReached == 1 ) || ( r.UserDefinedTermination == 1 ) )
					r.TerminateSimulation = 1;
			}
		}


		// FINALISATION -----------------------------------------------------------
		PerThread_Finalization(tid, __MPGOS_PERTHREAD_NT,r,SharedParameters,IntegerSharedParameters);


		// WRITE DATA BACK TO GLOBAL MEMORY ---------------------------------------
		//always
		GlobalVariables.d_ActualTime[tid]       = r.ActualTime;
		#pragma unroll
		for (int i=0; i<2; i++)
			GlobalVariables.d_TimeDomain[tid + i*__MPGOS_PERTHREAD_NT] = r.TimeDomain[i];

		#pragma unroll
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
			GlobalVariables.d_ActualState[tid + i*__MPGOS_PERTHREAD_NT] = r.ActualState[i];

		//if control parameters
		#if __MPGOS_PERTHREAD_NCP > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NCP; i++)
				GlobalVariables.d_ControlParameters[tid + i*__MPGOS_PERTHREAD_NT] = r.ControlParameters[i];
		#endif

		//if accessories
		#if __MPGOS_PERTHREAD_NA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NA; i++)
				GlobalVariables.d_Accessories[tid + i*__MPGOS_PERTHREAD_NT] = r.Accessories[i];
		#endif

		//if integere accessories
		#if __MPGOS_PERTHREAD_NIA > 0
			#pragma unroll
			for (int i=0; i<__MPGOS_PERTHREAD_NIA; i++)
				GlobalVariables.d_IntegerAccessories[tid + i*__MPGOS_PERTHREAD_NT] = r.IntegerAccessories[i];
		#endif

		#if __MPGOS_PERTHREAD_NDO > 0
			GlobalVariables.d_DenseOutputIndex[tid] = r.DenseOutputIndex;
		#endif
	}
}

#endif

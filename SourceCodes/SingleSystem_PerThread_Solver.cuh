#ifndef SINGLESYSTEM_PERTHREAD_SOLVER_H
#define SINGLESYSTEM_PERTHREAD_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "SingleSystem_PerThread_DenseOutput.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_Steppers.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_ErrorControllers.cuh"
#include "SingleSystem_PerThread_EventHandling.cuh"

__device__ void SharedStructLoad(SharedStruct &, Struct_GlobalVariables);

__global__ void SingleSystem_PerThread(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int tid = threadIdx.x + blockIdx.x*blockDim.x;


	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	__shared__ SharedStruct SharedSettings;
	SharedStructLoad(SharedSettings, GlobalVariables);
	SharedParametersStruct SharedMemoryPointers(GlobalVariables,SharedMemoryUsage);


	if (tid < ThreadConfiguration.NumberOfActiveThreads)
	{
		// REGISTER MEMORY MANAGEMENT -----------------------------------------
		RegisterStruct r(GlobalVariables,SolverOptions,tid);


		// INITIALISATION -----------------------------------------------------
		PerThread_Initialization(tid,__MPGOS_PERTHREAD_NT,r,SharedMemoryPointers);

		#if __MPGOS_PERTHREAD_NE > 0
			PerThread_EventFunction(tid,__MPGOS_PERTHREAD_NT,r.ActualEventValue,r,SharedMemoryPointers);
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
				PerThread_Stepper_RK4(tid,r,SharedMemoryPointers);
				PerThread_ErrorController_RK4(tid,r,SolverOptions.InitialTimeStep);
			#endif

			#if __MPGOS_PERTHREAD_ALGORITHM == 1
				PerThread_Stepper_RKCK45(tid,r,SharedMemoryPointers);
				PerThread_ErrorController_RKCK45(tid,r,SharedSettings,SolverOptions);
			#endif


			// NEW EVENT VALUE AND TIME STEP CONTROL---------------------------
			#if __MPGOS_PERTHREAD_NE > 0
				r.NewTimeStepTmp = r.ActualTime+r.TimeStep;
				PerThread_EventFunction(tid,__MPGOS_PERTHREAD_NT,r.ActualEventValue,r,SharedMemoryPointers);
				PerThread_EventTimeStepControl(tid,r,SharedSettings,SolverOptions.MinimumTimeStep);
			#endif


			// UPDATE PROCESS -------------------------------------------------
			if ( r.UpdateStep == 1 )
			{
				r.ActualTime += r.TimeStep;

				for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
					r.ActualState[i] = r.NextState[i];

				PerThread_ActionAfterSuccessfulTimeStep(tid,__MPGOS_PERTHREAD_NT,r,SharedMemoryPointers);

				#if __MPGOS_PERTHREAD_NE > 0
					for (int i=0; i<__MPGOS_PERTHREAD_NE; i++)
					{
						if ( ( ( r.ActualEventValue[i] >  SharedSettings.EventTolerance[i] ) && ( abs(r.NextEventValue[i]) < SharedSettings.EventTolerance[i] ) && ( SharedSettings.EventDirection[i] <= 0 ) ) || \
							 ( ( r.ActualEventValue[i] < -SharedSettings.EventTolerance[i] ) && ( abs(r.NextEventValue[i]) < SharedSettings.EventTolerance[i] ) && ( SharedSettings.EventDirection[i] >= 0 ) ) )
						{
							PerThread_ActionAfterEventDetection(tid,__MPGOS_PERTHREAD_NT,i,r,SharedMemoryPointers);
						}
					}

					PerThread_EventFunction(tid,__MPGOS_PERTHREAD_NT,r.ActualEventValue,r,SharedMemoryPointers);

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
		PerThread_Finalization(tid, __MPGOS_PERTHREAD_NT,r,SharedMemoryPointers);


		// WRITE DATA BACK TO GLOBAL MEMORY ---------------------------------------
		r.WriteToGlobalVariables(GlobalVariables,tid);
	}
}


__device__ void SharedStructLoad(SharedStruct &SharedSettings, Struct_GlobalVariables GlobalVariables)
{
	// Initialise tolerances of adaptive solvers
	#if __MPGOS_PERTHREAD_ADAPTIVE
		const int LaunchesSD = __MPGOS_PERTHREAD_SD / blockDim.x + (__MPGOS_PERTHREAD_SD % blockDim.x == 0 ? 0 : 1);
		#pragma unroll
		for (int j=0; j<LaunchesSD; j++)
		{
			int ltid = threadIdx.x + j*blockDim.x;

			if ( ltid < __MPGOS_PERTHREAD_SD)
			{
				SharedSettings.RelativeTolerance[ltid] = GlobalVariables.d_RelativeTolerance[ltid];
				SharedSettings.AbsoluteTolerance[ltid] = GlobalVariables.d_AbsoluteTolerance[ltid];
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
				SharedSettings.EventTolerance[ltid] = GlobalVariables.d_EventTolerance[ltid];
				SharedSettings.EventDirection[ltid] = GlobalVariables.d_EventDirection[ltid];
			}
		}
	#endif
}


#endif

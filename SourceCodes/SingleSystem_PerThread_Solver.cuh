#ifndef SINGLESYSTEM_PERTHREAD_SOLVER_H
#define SINGLESYSTEM_PERTHREAD_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "SingleSystem_PerThread_DenseOutput.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_Steppers.cuh"
#include "SingleSystem_PerThread_ExplicitRungeKutta_ErrorControllers.cuh"
#include "SingleSystem_PerThread_EventHandling.cuh"


template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
__global__ void SingleSystem_PerThread(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables<Precision> GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions<Precision> SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	
	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    MINIMUM ALLOCABLE MEMORY IS 1
	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;
	
	Precision* gs_SharedParameters = (Precision*)&DynamicSharedMemory;
		MemoryShift = (SharedMemoryUsage.PreferSharedMemory  == 1 ? NSP : 0);
	
	int* gs_IntegerSharedParameters = (int*)&gs_SharedParameters[MemoryShift];
	
	const bool IsAdaptive  = ( Algorithm==RK4 ? 0 : 1 );
	
	__shared__ Precision s_RelativeTolerance[ (IsAdaptive==0 ? 1 : SD) ];
	__shared__ Precision s_AbsoluteTolerance[ (IsAdaptive==0 ? 1 : SD) ];
	__shared__ Precision s_EventTolerance[ (NE==0 ? 1 : NE) ];
	__shared__ int       s_EventDirection[ (NE==0 ? 1 : NE) ];
	
	// Initialise tolerances of adaptive solvers
	if ( IsAdaptive  == 1 )
	{
		const int LaunchesSD = SD / blockDim.x + (SD % blockDim.x == 0 ? 0 : 1);
		#pragma unroll
		for (int j=0; j<LaunchesSD; j++)
		{
			int ltid = threadIdx.x + j*blockDim.x;
			
			if ( ltid < SD)
			{
				s_RelativeTolerance[ltid] = GlobalVariables.d_RelativeTolerance[ltid];
				s_AbsoluteTolerance[ltid] = GlobalVariables.d_AbsoluteTolerance[ltid];
			}
		}
	}
	
	// Initialise shared event handling variables
	const int LaunchesNE = NE / blockDim.x + (NE % blockDim.x == 0 ? 0 : 1);
	#pragma unroll
	for (int j=0; j<LaunchesNE; j++)
	{
		int ltid = threadIdx.x + j*blockDim.x;
		
		if ( ltid < SD)
		{
			s_EventTolerance[ltid] = GlobalVariables.d_EventTolerance[ltid];
			s_EventDirection[ltid] = GlobalVariables.d_EventDirection[ltid];
		}
	}
	
	// Initialise shared parameters
	if ( SharedMemoryUsage.PreferSharedMemory == 0 )
	{
		gs_SharedParameters        = GlobalVariables.d_SharedParameters;
		gs_IntegerSharedParameters = GlobalVariables.d_IntegerSharedParameters;
	} else
	{
		const int MaxElementNumber = max( NSP, NISP );
		const int LaunchesSP       = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);
		
		#pragma unroll
		for (int i=0; i<LaunchesSP; i++)
		{
			int ltid = threadIdx.x + i*blockDim.x;
			
			if ( ltid < NSP )
				gs_SharedParameters[ltid] = GlobalVariables.d_SharedParameters[ltid];
			
			if ( ltid < NISP )
				gs_IntegerSharedParameters[ltid] = GlobalVariables.d_IntegerSharedParameters[ltid];
		}
	}
	
	
	if (tid < ThreadConfiguration.NumberOfActiveThreads)
	{
		// REGISTER MEMORY MANAGEMENT -----------------------------------------
		//    MINIMUM ALLOCABLE MEMORY IS 1
		Precision r_TimeDomain[2];
		Precision r_ActualState[SD];
		Precision r_NextState[SD];
		Precision r_Error[SD];
		Precision r_ControlParameters[ (NCP==0 ? 1 : NCP) ];
		Precision r_Accessories[ (NA==0 ? 1 : NA) ];
		int       r_IntegerAccessories[ (NIA==0 ? 1 : NIA) ];
		Precision r_ActualEventValue[ (NE==0 ? 1 : NE) ];
		Precision r_NextEventValue[ (NE==0 ? 1 : NE) ];
		Precision r_ActualTime;
		Precision r_TimeStep;
		Precision r_NewTimeStep;
		Precision r_DenseOutputActualTime;
		int       r_DenseOutputIndex;
		int       r_UpdateDenseOutput;
		int       r_NumberOfSkippedStores;
		int       r_IsFinite;
		int       r_TerminateSimulation;
		int       r_UserDefinedTermination;
		int       r_UpdateStep;
		int       r_EndTimeDomainReached;
		
		#pragma unroll
		for (int i=0; i<2; i++)
			r_TimeDomain[i] = GlobalVariables.d_TimeDomain[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<SD; i++)
			r_ActualState[i] = GlobalVariables.d_ActualState[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<NCP; i++)
			r_ControlParameters[i] = GlobalVariables.d_ControlParameters[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<NA; i++)
			r_Accessories[i] = GlobalVariables.d_Accessories[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<NIA; i++)
			r_IntegerAccessories[i] = GlobalVariables.d_IntegerAccessories[tid + i*NT];
		
		r_ActualTime             = GlobalVariables.d_ActualTime[tid];
		r_TimeStep               = SolverOptions.InitialTimeStep;
		r_NewTimeStep            = SolverOptions.InitialTimeStep;
		r_DenseOutputIndex       = GlobalVariables.d_DenseOutputIndex[tid];
		r_DenseOutputActualTime  = r_ActualTime;
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
		
		if ( NE > 0 ) // Eliminated at compile time if NE=0
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
		
		if ( NDO > 0 ) // Eliminated at compile time if NDO=0
		{
			PerThread_StoreDenseOutput<NT, SD, NDO, Precision>(\
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
				PerThread_Stepper_RK4<NT,SD,Precision>(\
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
				PerThread_Stepper_RKCK45<NT,SD,Precision>(\
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
				
				PerThread_ErrorController_RKCK45<SD,Precision>(\
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
			if ( NE > 0 ) // Eliminated at compile time if NE=0
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
				
				PerThread_EventTimeStepControl<NE,Precision>(\
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
				
				for (int i=0; i<SD; i++)
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
				
				if ( NE > 0 ) // Eliminated at compile time if NE=0
				{
					for (int i=0; i<NE; i++)
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
					
					for (int i=0; i<NE; i++)
						r_ActualEventValue[i] = r_NextEventValue[i];
				}
				
				if ( NDO > 0 ) //Eliminated at compile time if NDO=0
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
					
					PerThread_StoreDenseOutput<NT, SD, NDO, Precision>(\
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
		for (int i=0; i<SD; i++)
			GlobalVariables.d_ActualState[tid + i*NT] = r_ActualState[i];
		
		#pragma unroll
		for (int i=0; i<NCP; i++)
			GlobalVariables.d_ControlParameters[tid + i*NT] = r_ControlParameters[i];
		
		#pragma unroll
		for (int i=0; i<NA; i++)
			GlobalVariables.d_Accessories[tid + i*NT] = r_Accessories[i];
		
		#pragma unroll
		for (int i=0; i<NIA; i++)
			GlobalVariables.d_IntegerAccessories[tid + i*NT] = r_IntegerAccessories[i];
		
		GlobalVariables.d_ActualTime[tid]       = r_ActualTime;
		GlobalVariables.d_DenseOutputIndex[tid] = r_DenseOutputIndex;
	}
}

#endif
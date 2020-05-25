#ifndef SINGLESYSTEM_PERTHREAD_SOLVER_H
#define SINGLESYSTEM_PERTHREAD_SOLVER_H

//#include "SingleSystem_PerThread_RungeKutta_DenseOutput.cuh"         // Dependency: NDO (NumberOfDenseOutputs)
//#include "SingleSystem_PerThread_RungeKutta_Steppers.cuh"            // No specialised templates
//#include "SingleSystem_PerThread_RungeKutta_ErrorController.cuh"     // No specialised templates
//#include "SingleSystem_PerThread_RungeKutta_EventHandling.cuh"       // Dependency: NE (NumberOfEvents)


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
		// REGISTER MEMORY MANAGEMENT ---------------------------------------------
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
		
		
		
		
		
		
		/*PerThread_Initialization(tid, NT, ActualTime, TimeStep, TimeDomain, ActualState, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
		double NextDenseOutputTime = ActualTime;
		
		StoreDenseOutput<NDO>(KernelParameters, tid, ActualState, ActualTime, TimeDomain[1], DenseOutputIndex, UpdateDenseOutput, NextDenseOutputTime);
		
		PerThread_EventFunction(tid, NT, ActualEventValue, ActualState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
		
		EventHandlingInitialisation<NE>(EventCounter, EventEquilibriumCounter);
		
		
		while ( TerminateSimulation==0 )
		{
			UpdateRungeKuttaStep = 1;
			UpdateDenseOutput = 0;
			IsFinite = 1;
			
			TimeStep = fmin(TimeStep, TimeDomain[1]-ActualTime);
			DenseOutputTimeStepCorrection<NDO>(KernelParameters, tid, UpdateDenseOutput, DenseOutputIndex, NextDenseOutputTime, ActualTime, TimeStep);
			
			
			if ( Algorithm==RK4 )
			{
				RungeKuttaStepperRK4<NT,SD,Algorithm>(tid, ActualTime, TimeStep, ActualState, NextState, Error, IsFinite, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
				ErrorControllerRK4(tid, KernelParameters.InitialTimeStep, IsFinite, TerminateSimulation, NewTimeStep);
			}
			
			if ( Algorithm==RKCK45 )
			{
				RungeKuttaStepperRKCK45<NT,SD,Algorithm>(tid, ActualTime, TimeStep, ActualState, NextState, Error, IsFinite, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
				ErrorControllerRKCK45<SD>(KernelParameters, tid, TimeStep, NextState, Error, s_RelativeTolerance, s_AbsoluteTolerance, UpdateRungeKuttaStep, IsFinite, TerminateSimulation, NewTimeStep);
			}
			
			PerThread_EventFunction(tid, NT, NextEventValue, NextState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
			EventHandlingTimeStepControl<NE>(tid, ActualEventValue, NextEventValue, UpdateRungeKuttaStep, s_EventTolerance, s_EventDirection, TimeStep, NewTimeStep, KernelParameters.MinimumTimeStep);
			
			
			if ( UpdateRungeKuttaStep == 1 )
			{
				ActualTime += TimeStep;
				
				for (int i=0; i<SD; i++)
					ActualState[i] = NextState[i];
				
				EventHandlingUpdate<NE>(tid, NT, ActualEventValue, NextEventValue, EventCounter, EventEquilibriumCounter, TerminateSimulation, KernelParameters.MaxStepInsideEvent, \
                                        s_EventTolerance, s_EventDirection, s_EventStopCounter, \
										ActualTime, TimeStep, TimeDomain, ActualState, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
				
				NumberOfSuccessfulTimeStep++;
				if ( ( KernelParameters.MaximumNumberOfTimeSteps != 0 ) && ( NumberOfSuccessfulTimeStep == KernelParameters.MaximumNumberOfTimeSteps ) )
					TerminateSimulation = 1;
				
				PerThread_ActionAfterSuccessfulTimeStep(tid, NT, ActualTime, TimeStep, TimeDomain, ActualState, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
				
				StoreDenseOutput<NDO>(KernelParameters, tid, ActualState, ActualTime, TimeDomain[1], DenseOutputIndex, UpdateDenseOutput, NextDenseOutputTime);
				
				if ( ActualTime > ( TimeDomain[1] - KernelParameters.MinimumTimeStep*1.01 ) )
					TerminateSimulation = 1;
			}
			
			TimeStep = NewTimeStep;
		}
		
		PerThread_Finalization(tid, NT, ActualTime, TimeStep, TimeDomain, ActualState, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
		
		#pragma unroll
		for (int i=0; i<2; i++)
			KernelParameters.d_TimeDomain[tid + i*NT] = TimeDomain[i];
		
		#pragma unroll
		for (int i=0; i<SD; i++)
			KernelParameters.d_ActualState[tid + i*NT] = ActualState[i];
		
		#pragma unroll
		for (int i=0; i<NCP; i++)
			KernelParameters.d_ControlParameters[tid + i*NT] = ControlParameters[i];
		
		#pragma unroll
		for (int i=0; i<NA; i++)
			KernelParameters.d_Accessories[tid + i*NT] = Accessories[i];
		
		#pragma unroll
		for (int i=0; i<NIA; i++)
			KernelParameters.d_IntegerAccessories[tid + i*NT] = IntegerAccessories[i];*/
	}
}


#endif
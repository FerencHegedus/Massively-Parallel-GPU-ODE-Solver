#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_H


__constant__ double d_BT_RK4[1];
__constant__ double d_BT_RKCK45[26];

#include "SingleSystem_PerThread_RungeKutta_LoadSharedVariables.cuh" // Dependency: Algorithm
#include "SingleSystem_PerThread_RungeKutta_DenseOutput.cuh"         // Dependency: NDO (NumberOfDenseOutputs)
#include "SingleSystem_PerThread_RungeKutta_Steppers.cuh"            // No specialised templates
#include "SingleSystem_PerThread_RungeKutta_ErrorController.cuh"     // No specialised templates
#include "SingleSystem_PerThread_RungeKutta_EventHandling.cuh"       // Dependency: NE (NumberOfEvents)


template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
__global__ void SingleSystem_PerThread_RungeKutta(IntegratorInternalVariables KernelParameters)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	extern __shared__ int DynamicSharedMemory[];
	double* s_SharedParameters        = nullptr;
	double* s_RelativeTolerance       = nullptr;
	double* s_AbsoluteTolerance       = nullptr;
	double* s_EventTolerance          = nullptr;
	int*    s_EventDirection          = nullptr;
	int*    s_EventStopCounter        = nullptr;
	int*    s_IntegerSharedParameters = nullptr;
	
	LoadSharedVariables<Algorithm>(DynamicSharedMemory, KernelParameters, s_SharedParameters, s_IntegerSharedParameters, s_RelativeTolerance, s_AbsoluteTolerance, s_EventTolerance, s_EventDirection, s_EventStopCounter);
	__syncthreads();
	
	if (tid < KernelParameters.ActiveThreads)
	{
		double TimeDomain[2];
		double ActualState[SD];
		double ControlParameters[ (NCP==0 ? 1 : NCP) ];
		double Accessories[ (NA==0 ? 1 : NA) ];
		int    IntegerAccessories[ (NIA==0 ? 1 : NIA) ];
		
		#pragma unroll
		for (int i=0; i<2; i++)
			TimeDomain[i] = KernelParameters.d_TimeDomain[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<SD; i++)
			ActualState[i] = KernelParameters.d_ActualState[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<NCP; i++)
			ControlParameters[i] = KernelParameters.d_ControlParameters[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<NA; i++)
			Accessories[i] = KernelParameters.d_Accessories[tid + i*NT];
		
		#pragma unroll
		for (int i=0; i<NIA; i++)
			IntegerAccessories[i] = KernelParameters.d_IntegerAccessories[tid + i*NT];
		
		
		double NextState[SD];
		double Error[ (Algorithm==RKCK45 ? SD : 1) ]; // Algorithm dependent !!!
		
		double ActualEventValue[ (NE==0 ? 1 : NE) ];
		double NextEventValue[ (NE==0 ? 1 : NE) ];
		int    EventCounter[ (NE==0 ? 1 : NE) ];
		int    EventEquilibriumCounter[ (NE==0 ? 1 : NE) ];
		
		double TimeStep   = KernelParameters.InitialTimeStep;
		double ActualTime = TimeDomain[0];
		
		int    NumberOfSuccessfulTimeStep = 0;
		bool   TerminateSimulation        = 0;
		int    DenseOutputIndex           = 0;
		bool   UpdateDenseOutput          = 1;
		
		bool   UpdateRungeKuttaStep;
		bool   IsFinite;
		double NewTimeStep;
		
		
		PerThread_Initialization(tid, NT, ActualTime, TimeStep, TimeDomain, ActualState, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
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
			KernelParameters.d_IntegerAccessories[tid + i*NT] = IntegerAccessories[i];
	}
}


#endif
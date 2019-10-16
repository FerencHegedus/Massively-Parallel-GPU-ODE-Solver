#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_H

__constant__ double d_BT_RK4[1];
__constant__ double d_BT_RKCK45[26];

#include "SingleSystem_PerThread_RungeKutta_LoadSharedVariables.cuh" // Dependency: SelectedAlgorithm
#include "SingleSystem_PerThread_RungeKutta_Steppers.cuh"            // Dependency: SelectedAlgorithm
#include "SingleSystem_PerThread_RungeKutta_ErrorController.cuh"     // Dependency: SelectedAlgorithm
#include "SingleSystem_PerThread_RungeKutta_DenseOutput.cuh"         // Dependency: SelectedDenseOutput
#include "SingleSystem_PerThread_RungeKutta_EventHandling.cuh"       // Dependency: SelectedEventHandling


template <AlgorithmOptions SelectedAlgorithm, EventHandlingOptions SelectedEventHandling, DenseOutputOptions SelectedDenseOutput>
__global__ void SingleSystem_PerThread_Runge_Kutta(IntegratorInternalVariables KernelParameters)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	extern __shared__ int DynamicSharedMemory[];
	double* s_SharedParameters = nullptr;
	double* s_RelativeTolerance = nullptr;
	double* s_AbsoluteTolerance = nullptr;
	double* s_EventTolerance = nullptr;
	int* s_EventDirection = nullptr;
	int* s_EventStopCounter = nullptr;
	int* s_IntegerSharedParameters = nullptr;
	
	LoadSharedVariables<SelectedAlgorithm>(DynamicSharedMemory, KernelParameters, \
	                                       s_SharedParameters, s_IntegerSharedParameters, \
	                                       s_RelativeTolerance, s_AbsoluteTolerance, \
										   s_EventTolerance, s_EventDirection, s_EventStopCounter);
	__syncthreads();
	
	if (tid < KernelParameters.ActiveThreads)
	{
		bool TerminateSimulation = 0;
		
		double ActualTime = KernelParameters.d_TimeDomain[tid];
		double NextDenseOutputTime = KernelParameters.d_TimeDomain[tid];
		double UpperTimeDomain = KernelParameters.d_TimeDomain[tid + KernelParameters.NumberOfThreads];
		
		int  DenseOutputIndex  = 0;
		bool UpdateDenseOutput = 1;
		StoreDenseOutput<SelectedDenseOutput>(KernelParameters, tid, ActualTime, UpperTimeDomain, DenseOutputIndex, UpdateDenseOutput, NextDenseOutputTime);
		
		PerThread_EventFunction(tid, KernelParameters.NumberOfThreads, KernelParameters.d_ActualEventValue, \
		                        KernelParameters.d_ActualState, ActualTime, \
								KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
		
		PerThread_Initialization(tid, KernelParameters.NumberOfThreads, \
		                         ActualTime, KernelParameters.InitialTimeStep, KernelParameters.d_TimeDomain, KernelParameters.d_ActualState, \
								 KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
		
		EventHandlingInitialisation<SelectedEventHandling>(KernelParameters, tid);
		
		bool UpdateRungeKuttaStep;
		bool IsFinite;
		double TimeStep = KernelParameters.InitialTimeStep;
		double NewTimeStep;
		int TemporaryIndex;
		int NumberOfSuccessfulTimeStep = 0;
		while ( TerminateSimulation==0 )
		{
			UpdateRungeKuttaStep = 1;
			UpdateDenseOutput = 0;
			IsFinite = 1;
			
			TimeStep = fmin(TimeStep, UpperTimeDomain-ActualTime);
			
			DenseOutputTimeStepCorrection<SelectedDenseOutput>(KernelParameters, tid, UpdateDenseOutput, DenseOutputIndex, NextDenseOutputTime, ActualTime, TimeStep);
			
			RungeKuttaStepper<SelectedAlgorithm>(KernelParameters, tid, ActualTime, TimeStep, s_SharedParameters, s_IntegerSharedParameters, IsFinite);
			
			ErrorController<SelectedAlgorithm>(KernelParameters, tid, s_RelativeTolerance, s_AbsoluteTolerance, \
			                                   UpdateRungeKuttaStep, IsFinite, TerminateSimulation, TimeStep, NewTimeStep);
			
			PerThread_EventFunction(tid, KernelParameters.NumberOfThreads, KernelParameters.d_NextEventValue, \
			                        KernelParameters.d_NextState, ActualTime, \
									KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
			
			EventHandlingTimeStepControl<SelectedEventHandling>(KernelParameters, tid, UpdateRungeKuttaStep, \
                                                                s_EventTolerance, s_EventDirection, TimeStep, NewTimeStep);
			
			if ( UpdateRungeKuttaStep == 1 )
			{
				ActualTime += TimeStep;
				
				TemporaryIndex = tid;
				for (int i=0; i<KernelParameters.SystemDimension; i++)
				{
					KernelParameters.d_ActualState[TemporaryIndex] = KernelParameters.d_NextState[TemporaryIndex];
					TemporaryIndex += KernelParameters.NumberOfThreads;
				}
				
				EventHandlingUpdate<SelectedEventHandling>(KernelParameters, tid, TerminateSimulation, \
                                                           s_EventTolerance, s_EventDirection, s_EventStopCounter, \
														   ActualTime, TimeStep, s_SharedParameters, s_IntegerSharedParameters);
				
				NumberOfSuccessfulTimeStep++;
				if ( ( KernelParameters.MaximumNumberOfTimeSteps != 0 ) && ( NumberOfSuccessfulTimeStep == KernelParameters.MaximumNumberOfTimeSteps ) )
					TerminateSimulation = 1;
				
				PerThread_ActionAfterSuccessfulTimeStep(tid, KernelParameters.NumberOfThreads, \
				                                        ActualTime, TimeStep, KernelParameters.d_TimeDomain, KernelParameters.d_ActualState, \
														KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
				
				StoreDenseOutput<SelectedDenseOutput>(KernelParameters, tid, ActualTime, UpperTimeDomain, DenseOutputIndex, UpdateDenseOutput, NextDenseOutputTime);
				
				if ( ActualTime > ( UpperTimeDomain - KernelParameters.MinimumTimeStep*1.01 ) )
					TerminateSimulation = 1;
			}
			
			TimeStep = NewTimeStep;
		}
		
		PerThread_Finalization(tid, KernelParameters.NumberOfThreads, \
		                       ActualTime, TimeStep, KernelParameters.d_TimeDomain, KernelParameters.d_ActualState, \
							   KernelParameters.d_ControlParameters, s_SharedParameters, s_IntegerSharedParameters, KernelParameters.d_Accessories, KernelParameters.d_IntegerAccessories);
	}
}

#endif
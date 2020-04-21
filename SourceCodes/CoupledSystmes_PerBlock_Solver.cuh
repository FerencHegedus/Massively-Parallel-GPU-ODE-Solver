#ifndef COUPLEDSYSTEMS_PERBLOCK_SOLVER_H
#define COUPLEDSYSTEMS_PERBLOCK_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "CoupledSystems_PerBlock_ExplicitRungeKutta_Steppers.cuh"         // No specialised templates
#include "CoupledSystems_PerBlock_ExplicitRungeKutta_ErrorController.cuh"  // No specialised templates

template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
__global__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables<Precision> GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions<Precision> SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LogicalThreadsPerBlock = SPB * UPS;
	const int NumberOfBlockLaunches  = LogicalThreadsPerBlock / TPB + (LogicalThreadsPerBlock % TPB == 0 ? 0 : 1);
	const int ThreadPaddingPerBlock  = NumberOfBlockLaunches * TPB - LogicalThreadsPerBlock;
	const int TotalLogicalThreads         = (LogicalThreadsPerBlock + ThreadPaddingPerBlock) * gridDim.x;
	const int TotalLogicalThreadsPerBlock = (LogicalThreadsPerBlock + ThreadPaddingPerBlock);
	
	const int GlobalThreadID_GPU = threadIdx.x + blockIdx.x*blockDim.x;
	const int LocalThreadID_GPU  = threadIdx.x;
	const int BlockID            = blockIdx.x;
	
	int GlobalThreadID_Logical; // Depends on BlockLaunch
	int LocalThreadID_Logical;  // Depends on BlockLaunch
	int GlobalMemoryID;         // Depends on BlockLaunch
	int LocalMemoryID;          // Depends on BlockLaunch
	
	int GlobalSystemID;         // Depends on BlockLaunch
	int LocalSystemID;          // Depends on BlockLaunch
	int UnitID;                 // Depends on BlockLaunch
	
	
	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    SYSTEMS ARE PADDED IN SHARED MEMORY (NO GLOBAL BOUND CHECK IS NECESSARY AFTER LOADING)
	//    MINIMUM ALLOCABLE MEMORY IS 1
	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;
	Precision* gs_GlobalParameters        = (Precision*)&DynamicSharedMemory;               MemoryShift = (SharedMemoryUsage.GlobalVariables  == 1 ? NGP : 0);
	Precision* gs_CouplingMatrix          = (Precision*)&gs_GlobalParameters[MemoryShift];  MemoryShift = (SharedMemoryUsage.CouplingMatrices == 1 ? NC*UPS*UPS : 0);
	int*       gs_IntegerGlobalParameters = (int*)&gs_CouplingMatrix[MemoryShift];
	
	const bool IsAdaptive = ( Algorithm==RK4 ? 0 : 1 );
	//const bool NCpadding  = 
	__shared__ Precision s_CouplingTerms[SPB][UPS][NC];                                       // Need access by user
	__shared__ Precision s_CouplingStrength[SPB][NC];                                         // Internal
	__shared__ Precision s_TimeDomain[SPB][2];                                                // Need access by user
	__shared__ Precision s_ActualTime[SPB];                                                   // Need access by user
	__shared__ Precision s_TimeStep[SPB];                                                     // Need access by user
	__shared__ Precision s_NewTimeStep[SPB];                                                  // Need access by user
	__shared__ Precision s_SystemParameters[ (NSP==0 ? 1 : SPB) ][ (NSP==0 ? 1 : NSP) ];      // Need access by user
	__shared__ Precision s_SystemAccessories[ (NSA==0 ? 1 : SPB) ][ (NSA==0 ? 1 : NSA) ];     // Need access by user
	__shared__ Precision s_RelativeTolerance[ (IsAdaptive==0 ? 1 : UD) ];                     // Internal
	__shared__ Precision s_AbsoluteTolerance[ (IsAdaptive==0 ? 1 : UD) ];                     // Internal
	__shared__ Precision s_EventTolerance[ (NE==0 ? 1 : NE) ];                                // Internal
	__shared__ int s_CouplingIndex[NC];                                                       // Internal
	__shared__ int s_DenseOutputIndex[SPB];                                                   // Internal
	__shared__ int s_IntegerSystemAccessories[ (NiSA==0 ? 1 : SPB) ][ (NiSA==0 ? 1 : NiSA) ]; // Need access by user
	__shared__ int s_EventDirection[ (NE==0 ? 1 : NE) ];                                      // Need access by user
	__shared__ int s_EventStopCounter[ (NE==0 ? 1 : NE) ];                                    // Need access by user
	__shared__ int s_TerminatedSystemsPerBlock;                                               // Internal
	__shared__ int s_IsFinite[SPB];                                                           // Internal
	__shared__ int s_TerminateSystemScope[SPB];                                               // Internal
	__shared__ int s_UpdateStep[SPB];                                                         // Internal
	__shared__ int s_EndTimeDomainReached[SPB];                                               // Internal
	__shared__ int s_NumberOfSuccessfulTimeStep[SPB];                                         // Internal
	
	// Initialise block scope variables
	if ( LocalThreadID_GPU == 0 )
		s_TerminatedSystemsPerBlock = 0;
	__syncthreads();
	
	// Initialise system scope variables
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		LocalSystemID  = threadIdx.x    + j*blockDim.x;
		GlobalSystemID = LocalSystemID  + BlockID*SPB;
		
		if ( ( LocalSystemID < SPB ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
		{
			for (int i=0; i<2; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				s_TimeDomain[LocalSystemID][i] = GlobalVariables.d_TimeDomain[GlobalMemoryID];
				
				if ( i==0 )
				{
					s_ActualTime[LocalSystemID]  = s_TimeDomain[LocalSystemID][i];
					s_TimeStep[LocalSystemID]    = SolverOptions.InitialTimeStep;
					s_NewTimeStep[LocalSystemID] = SolverOptions.InitialTimeStep;
					
					s_NumberOfSuccessfulTimeStep[LocalSystemID] = 0;
				}
			}
			
			for (int i=0; i<NSP; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				s_SystemParameters[LocalSystemID][i] = GlobalVariables.d_SystemParameters[GlobalMemoryID];
			}
			
			for (int i=0; i<NSA; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				s_SystemAccessories[LocalSystemID][i] = GlobalVariables.d_SystemAccessories[GlobalMemoryID];
			}
			
			for (int i=0; i<NiSA; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				s_IntegerSystemAccessories[LocalSystemID][i] = GlobalVariables.d_IntegerSystemAccessories[GlobalMemoryID];
			}
			
			for (int i=0; i<NC; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				s_CouplingStrength[LocalSystemID][i] = GlobalVariables.d_CouplingStrength[GlobalMemoryID];
			}
			
			for (int i=0; i<1; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				s_DenseOutputIndex[LocalSystemID] = GlobalVariables.d_DenseOutputIndex[GlobalMemoryID];
				s_TerminateSystemScope[LocalSystemID] = 0;
			}
		}
		
		if ( ( LocalSystemID < SPB ) && ( ( GlobalSystemID >= SolverOptions.ActiveSystems ) || ( GlobalSystemID >= NS ) ) )
		{
			s_TerminateSystemScope[LocalSystemID] = 1;
			s_UpdateStep[LocalSystemID] = 0;
			
			atomicAdd(&s_TerminatedSystemsPerBlock, 1);
		}
	}
	
	// Initialise solver tolerances
	if ( IsAdaptive  == 1 )
	{
		Launches = UD / blockDim.x + (UD % blockDim.x == 0 ? 0 : 1);
		
		for (int i=0; i<Launches; i++)
		{
			int idx = threadIdx.x + i*blockDim.x;
			
			if ( idx < UD )
			{
				s_RelativeTolerance[idx] = GlobalVariables.d_RelativeTolerance[idx];
				s_AbsoluteTolerance[idx] = GlobalVariables.d_AbsoluteTolerance[idx];
			}
		}
	}
	
	// Initialise event options
	Launches = NE / blockDim.x + (NE % blockDim.x == 0 ? 0 : 1);
	for (int i=0; i<Launches; i++)
	{
		int idx = threadIdx.x + i*blockDim.x;
		
		if ( idx < NE )
		{
			s_EventTolerance[idx]   = GlobalVariables.d_EventTolerance[idx];
			s_EventDirection[idx]   = GlobalVariables.d_EventDirection[idx];
			s_EventStopCounter[idx] = GlobalVariables.d_EventStopCounter[idx];
		}
	}
	
	// Initialise global scope variables if ON
	if ( SharedMemoryUsage.GlobalVariables  == 0 )
	{
		gs_GlobalParameters        = GlobalVariables.d_GlobalParameters;
		gs_IntegerGlobalParameters = GlobalVariables.d_IntegerGlobalParameters;
	} else
	{
		int MaxElementNumber = max( NGP, NiGP );
		Launches = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);
		
		for (int i=0; i<Launches; i++)
		{
			int idx = threadIdx.x + i*blockDim.x;
			
			if ( idx < NGP )
				gs_GlobalParameters[idx] = GlobalVariables.d_GlobalParameters[idx];
			
			if ( idx < NiGP )
				gs_IntegerGlobalParameters[idx] = GlobalVariables.d_IntegerGlobalParameters[idx];
		}
	}
	
	// Initialise coupling matrices if ON
	if ( SharedMemoryUsage.CouplingMatrices  == 0 )
	{
		gs_CouplingMatrix = GlobalVariables.d_CouplingMatrix;
	} else
	{
		int MaxElementNumber = NC*UPS*UPS;
		Launches = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);
		
		for (int i=0; i<Launches; i++)
		{
			int idx = threadIdx.x + i*blockDim.x;
			
			if ( idx < MaxElementNumber )
				gs_CouplingMatrix[idx] = GlobalVariables.d_CouplingMatrix[idx];
		}
	}
	
	// Initialise coupling index
	Launches = NC / blockDim.x + (NC % blockDim.x == 0 ? 0 : 1);	
	for (int i=0; i<Launches; i++)
	{
		int idx = threadIdx.x + i*blockDim.x;
		
		if ( idx < NC )
			s_CouplingIndex[idx] = GlobalVariables.d_CouplingIndex[idx];
	}
	
	
	// REGISTER MEMORY MANAGEMENT ---------------------------------------------
	//    THREADS ARE PADDED (NO GLOBAL BOUND CHECK IS NECESSARY AFTER LOADING)
	//    MINIMUM ALLOCABLE MEMORY IS 1
	Precision r_CouplingFactor[NumberOfBlockLaunches][NC];
	Precision r_ActualState[NumberOfBlockLaunches][UD];
	Precision r_NextState[NumberOfBlockLaunches][UD];
	Precision r_Error[NumberOfBlockLaunches][UD];
	Precision r_UnitParameters[ (NUP==0 ? 1 : NumberOfBlockLaunches) ][ (NUP==0 ? 1 : NUP) ];
	Precision r_UnitAccessories[ (NUA==0 ? 1 : NumberOfBlockLaunches) ][ (NUA==0 ? 1 : NUA) ];
	int       r_IntegerUnitAccessories[ (NiUA==0 ? 1 : NumberOfBlockLaunches) ][ (NiUA==0 ? 1 : NiUA) ];
	
	// Initialise unit scope variables
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		LocalThreadID_Logical  = LocalThreadID_GPU + BL*blockDim.x;
		GlobalThreadID_Logical = LocalThreadID_Logical + BlockID*TotalLogicalThreadsPerBlock;
		
		for (int i=0; i<UD; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			r_ActualState[BL][i] = GlobalVariables.d_ActualState[GlobalMemoryID];
		}
		
		for (int i=0; i<NUP; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			r_UnitParameters[BL][i] = GlobalVariables.d_UnitParameters[GlobalMemoryID];
		}
		
		for (int i=0; i<NUA; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			r_UnitAccessories[BL][i] = GlobalVariables.d_UnitAccessories[GlobalMemoryID];
		}
		
		for (int i=0; i<NiUA; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			r_IntegerUnitAccessories[BL][i] = GlobalVariables.d_IntegerUnitAccessories[GlobalMemoryID];
		}
	}
	__syncthreads();
	
	
	// SOLVER MANAGEMENT ------------------------------------------------------
	while ( s_TerminatedSystemsPerBlock < SPB )
	{
		// INITIALISE TIME STEPPING -------------------------------------------
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
			LocalSystemID         = LocalThreadID_Logical / UPS;
			UnitID                = LocalThreadID_Logical % UPS;
			
			if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) && ( UnitID == 0 ) )
			{
				s_UpdateStep[LocalSystemID]           = 1;
				s_IsFinite[LocalSystemID]             = 1;
				s_EndTimeDomainReached[LocalSystemID] = 0;
				
				s_TimeStep[LocalSystemID]   = s_NewTimeStep[LocalSystemID];
				
				if ( s_TimeStep[LocalSystemID] > ( s_TimeDomain[LocalSystemID][1] - s_ActualTime[LocalSystemID] ) )
				{
					s_TimeStep[LocalSystemID] = s_TimeDomain[LocalSystemID][1] - s_ActualTime[LocalSystemID];
					s_TimeStep[LocalSystemID] = MPGOS::FMAX(s_TimeStep[LocalSystemID], SolverOptions.MinimumTimeStep);
					
					s_EndTimeDomainReached[LocalSystemID] = 1;
				}
			}
		}
		__syncthreads();
		
		
		// STEPPER ------------------------------------------------------------
		if ( Algorithm == RK4 ) // Resolved at compile time as Algorithm is a template parameter and RK4 is constant
		{
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_RK4<NumberOfBlockLaunches, NS, UPS, UD, TPB, SPB, NC, NUP, NSP, NGP, NiGP, NUA, NiUA, NSA, NiSA, NE, NDO, Precision>( \
				r_ActualState, \
				r_NextState, \
				s_ActualTime, \
				s_TimeStep, \
				s_IsFinite, \
				r_UnitParameters, \
				s_SystemParameters, \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				r_UnitAccessories, \
				r_IntegerUnitAccessories, \
				s_SystemAccessories, \
				s_IntegerSystemAccessories, \
				s_CouplingTerms, \
				r_CouplingFactor, \
				gs_CouplingMatrix, \
				s_CouplingStrength, \
				s_CouplingIndex);
			
			MultipleSystems_MultipleBlockLaunches_ErrorController_RK4<NumberOfBlockLaunches, UPS, SPB, Precision>( \
				s_IsFinite, \
				s_NewTimeStep, \
				SolverOptions.InitialTimeStep, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep);
		}
		
		if ( Algorithm == RKCK45 ) // Resolved at compile time as Algorithm is a template parameter and RKCK45 is constant
		{
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_RKCK45<NumberOfBlockLaunches, NS, UPS, UD, TPB, SPB, NC, NUP, NSP, NGP, NiGP, NUA, NiUA, NSA, NiSA, NE, NDO, Precision>( \
				r_ActualState, \
				r_NextState, \
				s_ActualTime, \
				s_TimeStep, \
				s_IsFinite, \
				r_Error, \
				r_UnitParameters, \
				s_SystemParameters, \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				r_UnitAccessories, \
				r_IntegerUnitAccessories, \
				s_SystemAccessories, \
				s_IntegerSystemAccessories, \
				s_CouplingTerms, \
				r_CouplingFactor, \
				gs_CouplingMatrix, \
				s_CouplingStrength, \
				s_CouplingIndex);
			
			MultipleSystems_MultipleBlockLaunches_ErrorController_RKCK45<NumberOfBlockLaunches, UPS, UD, SPB, Precision>( \
				s_IsFinite, \
				s_TimeStep, \
				s_NewTimeStep, \
				r_NextState, \
				r_Error, \
				s_RelativeTolerance, \
				s_AbsoluteTolerance, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep, \
				SolverOptions);
		}
		
		
		// UPDATE PROCESS -----------------------------------------------------
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
			LocalSystemID         = LocalThreadID_Logical / UPS;
			UnitID                = LocalThreadID_Logical % UPS;
			GlobalSystemID        = LocalSystemID + BlockID*SPB;
			
			if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
			{
				if ( UnitID == 0 )
				{
					s_ActualTime[LocalSystemID] += s_TimeStep[LocalSystemID];
					
					if ( s_EndTimeDomainReached[LocalSystemID] == 1 )
					{
						s_TerminateSystemScope[LocalSystemID] = 1;
						atomicAdd(&s_TerminatedSystemsPerBlock, 1);
						
						s_UpdateStep[LocalSystemID]           = 0;
					}
					
					s_NumberOfSuccessfulTimeStep[LocalSystemID]++;
					if ( ( SolverOptions.MaximumNumberOfTimeSteps != 0 ) && ( s_NumberOfSuccessfulTimeStep[LocalSystemID] == SolverOptions.MaximumNumberOfTimeSteps ) )
					{
						printf("Warning: Maximum number of allowed time steps is reachd! (global system id: %d, @time: %+6.5e \n", GlobalSystemID, s_ActualTime[LocalSystemID]);
						s_TerminateSystemScope[LocalSystemID] = 1;
						atomicAdd(&s_TerminatedSystemsPerBlock, 1);
					}
				}
				
				for (int i=0; i<UD; i++)
					r_ActualState[BL][i] = r_NextState[BL][i];
			}
		}
		__syncthreads();
	}
	__syncthreads();
	
	
	// WRITE DATA BACK TO GLOBAL MEMORY ---------------------------------------
	
	// Unit scope variables (register variables)
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		LocalThreadID_Logical  = LocalThreadID_GPU + BL*blockDim.x;
		GlobalThreadID_Logical = LocalThreadID_Logical + BlockID*TotalLogicalThreadsPerBlock;
		
		for (int i=0; i<UD; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			GlobalVariables.d_ActualState[GlobalMemoryID] = r_ActualState[BL][i];
		}
		
		for (int i=0; i<NUA; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			GlobalVariables.d_UnitAccessories[GlobalMemoryID] = r_UnitAccessories[BL][i];
		}
		
		for (int i=0; i<NiUA; i++)
		{
			GlobalMemoryID = GlobalThreadID_Logical + i*TotalLogicalThreads;
			GlobalVariables.d_IntegerUnitAccessories[GlobalMemoryID] = r_IntegerUnitAccessories[BL][i];
		}
	}
	__syncthreads();
	
	// System scope variables (shared variables)
	Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		LocalSystemID  = threadIdx.x    + j*blockDim.x;
		GlobalSystemID = LocalSystemID  + BlockID*SPB;
		
		if ( ( LocalSystemID < SPB ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
		{
			for (int i=0; i<2; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				GlobalVariables.d_TimeDomain[GlobalMemoryID] = s_TimeDomain[LocalSystemID][i];
			}
			
			for (int i=0; i<NSA; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				GlobalVariables.d_SystemAccessories[GlobalMemoryID] = s_SystemAccessories[LocalSystemID][i];
			}
			
			for (int i=0; i<NiSA; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				GlobalVariables.d_IntegerSystemAccessories[GlobalMemoryID] = s_IntegerSystemAccessories[LocalSystemID][i];
			}
			
			for (int i=0; i<1; i++)
			{
				GlobalMemoryID = GlobalSystemID + i*NS;
				GlobalVariables.d_DenseOutputIndex[GlobalMemoryID] = s_DenseOutputIndex[LocalSystemID];
			}
		}
	}
	__syncthreads();
}

#endif
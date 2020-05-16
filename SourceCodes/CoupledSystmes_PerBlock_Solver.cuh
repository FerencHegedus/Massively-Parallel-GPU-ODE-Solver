#ifndef COUPLEDSYSTEMS_PERBLOCK_SOLVER_H
#define COUPLEDSYSTEMS_PERBLOCK_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "CoupledSystems_PerBlock_ExplicitRungeKutta_Steppers.cuh"
#include "CoupledSystems_PerBlock_ExplicitRungeKutta_ErrorController.cuh"
#include "CoupledSystems_PerBlock_EventHandling.cuh"
#include "CoupledSystems_PerBlock_DenseOutput.cuh"

template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
__global__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables<Precision> GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions<Precision> SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LogicalThreadsPerBlock      = SPB * UPS;
	const int NumberOfBlockLaunches       = LogicalThreadsPerBlock / TPB + (LogicalThreadsPerBlock % TPB == 0 ? 0 : 1);
	const int ThreadPaddingPerBlock       = NumberOfBlockLaunches * TPB - LogicalThreadsPerBlock;
	const int TotalLogicalThreads         = (LogicalThreadsPerBlock + ThreadPaddingPerBlock) * gridDim.x;
	const int TotalLogicalThreadsPerBlock = (LogicalThreadsPerBlock + ThreadPaddingPerBlock);
	const int LocalThreadID_GPU           = threadIdx.x;
	const int BlockID                     = blockIdx.x;
	
	int GlobalThreadID_Logical; // Depends on BlockLaunch
	int LocalThreadID_Logical;  // Depends on BlockLaunch
	int GlobalMemoryID;         // Depends on BlockLaunch
	int GlobalSystemID;         // Depends on BlockLaunch
	int LocalSystemID;          // Depends on BlockLaunch
	int UnitID;                 // Depends on BlockLaunch
	
	
	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    SYSTEMS ARE PADDED IN SHARED MEMORY (NO GLOBAL BOUND CHECK IS NECESSARY AFTER LOADING)
	//    MINIMUM ALLOCABLE MEMORY IS 1
	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;
	
	Precision* gs_GlobalParameters = (Precision*)&DynamicSharedMemory;
		MemoryShift = (SharedMemoryUsage.GlobalVariables  == 1 ? NGP : 0);
	
	Precision* gs_CouplingMatrix = (Precision*)&gs_GlobalParameters[MemoryShift];
		MemoryShift = (SharedMemoryUsage.CouplingMatrices == 1 ? NC*SharedMemoryUsage.SingleCouplingMatrixSize : 0);
	
	int* gs_IntegerGlobalParameters = (int*)&gs_CouplingMatrix[MemoryShift];
	
	const bool IsAdaptive  = ( Algorithm==RK4 ? 0 : 1 );
	
	// Bank conflict if NCmod = 0, 2, 4, 8 or 16
	const int NCmod        = NC % 32;
	const int IsPowerOfTwo = !( NCmod & (NCmod-1) ); // Including 0!
	const int NCp          = NC + ( NCmod==1 ? 0 : ( IsPowerOfTwo==1 ? 1 : 0 ) );
	// Bank conflicts if NSP, NSA and NiSA = 4, 8 or 16
	const int NSPp  = (  NSP==4 ?  NSP+1 : (  NSP==8 ?  NSP+1 : (  NSP==16 ?  NSP+1 : NSP  ) ) );
	const int NSAp  = (  NSA==4 ?  NSA+1 : (  NSA==8 ?  NSA+1 : (  NSA==16 ?  NSA+1 : NSA  ) ) );
	const int NiSAp = ( NiSA==4 ? NiSA+1 : ( NiSA==8 ? NiSA+1 : ( NiSA==16 ? NiSA+1 : NiSA ) ) );
	
	__shared__ Precision s_CouplingTerms[SPB][UPS][NCp];                                         // Need access by user
	__shared__ Precision s_CouplingStrength[SPB][NCp];                                           // Internal
	__shared__ Precision s_TimeDomain[SPB][2];                                                   // Need access by user
	__shared__ Precision s_ActualTime[SPB];                                                      // Need access by user
	__shared__ Precision s_TimeStep[SPB];                                                        // Need access by user
	__shared__ Precision s_NewTimeStep[SPB];                                                     // Need access by user
	__shared__ Precision s_SystemParameters[ (NSPp==0 ? 1 : SPB) ][ (NSPp==0 ? 1 : NSPp) ];      // Need access by user
	__shared__ Precision s_SystemAccessories[ (NSAp==0 ? 1 : SPB) ][ (NSAp==0 ? 1 : NSAp) ];     // Need access by user
	__shared__ Precision s_RelativeTolerance[ (IsAdaptive==0 ? 1 : UD) ];                        // Internal
	__shared__ Precision s_AbsoluteTolerance[ (IsAdaptive==0 ? 1 : UD) ];                        // Internal
	__shared__ Precision s_EventTolerance[ (NE==0 ? 1 : NE) ];                                   // Internal
	__shared__ Precision s_DenseOutputActualTime[SPB];                                           // Internal
	__shared__ int s_CouplingIndex[NC];                                                          // Internal
	__shared__ int s_DenseOutputIndex[SPB];                                                      // Internal
	__shared__ int s_UpdateDenseOutput[SPB];                                                     // Internal
	__shared__ int s_NumberOfSkippedStores[SPB];                                                 // Internal
	__shared__ int s_IntegerSystemAccessories[ (NiSAp==0 ? 1 : SPB) ][ (NiSAp==0 ? 1 : NiSAp) ]; // Need access by user
	__shared__ int s_EventDirection[ (NE==0 ? 1 : NE) ];                                         // Need access by user
	__shared__ int s_TerminatedSystemsPerBlock;                                                  // Internal
	__shared__ int s_IsFinite[SPB];                                                              // Internal
	__shared__ int s_TerminateSystemScope[SPB];                                                  // Internal
	__shared__ int s_UserDefinedTermination[SPB];                                                // Internal
	__shared__ int s_UpdateStep[SPB];                                                            // Internal
	__shared__ int s_EndTimeDomainReached[SPB];                                                  // Internal
	
	// Initialise block scope variables
	if ( LocalThreadID_GPU == 0 )
		s_TerminatedSystemsPerBlock = 0;
	__syncthreads();
	
	// Initialise system scope variables
	int Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
	for (int j=0; j<Launches; j++)
	{
		LocalSystemID  = threadIdx.x   + j*blockDim.x;
		GlobalSystemID = LocalSystemID + BlockID*SPB;
		
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
				s_DenseOutputIndex[LocalSystemID]       = GlobalVariables.d_DenseOutputIndex[GlobalMemoryID];
				s_DenseOutputActualTime[LocalSystemID]  = s_TimeDomain[LocalSystemID][0];
				s_UpdateDenseOutput[LocalSystemID]      = 1;
				s_NumberOfSkippedStores[LocalSystemID]  = 0;
				s_TerminateSystemScope[LocalSystemID]   = 0;
				s_UserDefinedTermination[LocalSystemID] = 0;
			}
		}
		
		if ( ( LocalSystemID < SPB ) && ( ( GlobalSystemID >= SolverOptions.ActiveSystems ) || ( GlobalSystemID >= NS ) ) )
		{
			atomicAdd(&s_TerminatedSystemsPerBlock, 1);
			s_TerminateSystemScope[LocalSystemID] = 1;
			s_UpdateStep[LocalSystemID]           = 0;
			s_UpdateDenseOutput[LocalSystemID]    = 0;
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
		int MaxElementNumber = NC*SharedMemoryUsage.SingleCouplingMatrixSize;
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
	Precision r_ActualEventValue[ (NE==0 ? 1 : NumberOfBlockLaunches) ][ (NE==0 ? 1 : NE) ];
	Precision r_NextEventValue[ (NE==0 ? 1 : NumberOfBlockLaunches) ][ (NE==0 ? 1 : NE) ];
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
	
	
	// INITIALISATION
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
		{
			CoupledSystems_PerBlock_Initialization<Precision>(\
				GlobalSystemID, \
				UnitID, \
				s_ActualTime[LocalSystemID], \
				s_TimeStep[LocalSystemID], \
				&s_TimeDomain[LocalSystemID][0], \
				&r_ActualState[BL][0], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0]);
		}
	}
	__syncthreads();
	
	if ( NE > 0 ) // Eliminated at compile time if NE=0
	{
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
			LocalSystemID         = LocalThreadID_Logical / UPS;
			UnitID                = LocalThreadID_Logical % UPS;
			GlobalSystemID        = LocalSystemID + BlockID*SPB;
			
			if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
			{
				CoupledSystems_PerBlock_EventFunction<Precision>(\
					GlobalSystemID, \
					UnitID, \
					&r_ActualEventValue[BL][0], \
					s_ActualTime[LocalSystemID], \
					s_TimeStep[LocalSystemID], \
					&s_TimeDomain[LocalSystemID][0], \
					&r_ActualState[BL][0], \
					&r_UnitParameters[BL][0], \
					&s_SystemParameters[LocalSystemID][0], \
					gs_GlobalParameters, \
					gs_IntegerGlobalParameters, \
					&r_UnitAccessories[BL][0], \
					&r_IntegerUnitAccessories[BL][0], \
					&s_SystemAccessories[LocalSystemID][0], \
					&s_IntegerSystemAccessories[LocalSystemID][0]);
			}
		}
		__syncthreads();
	}
	
	
	if ( NDO > 0 ) // Eliminated at compile time if NDO=0
	{
		CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_StoreDenseOutput<NumberOfBlockLaunches, NS, UPS, UD, SPB, Precision>(\
			s_UpdateStep, \
			s_UpdateDenseOutput, \
			s_DenseOutputIndex, \
			s_NumberOfSkippedStores, \
			GlobalVariables.d_DenseOutputTimeInstances, \
			GlobalVariables.d_DenseOutputStates, \
			s_DenseOutputActualTime, \
			s_ActualTime, \
			r_ActualState, \
			s_TimeDomain, \
			ThreadConfiguration, \
			SolverOptions);
	}
	
	
	// SOLVER MANAGEMENT ------------------------------------------------------
	while ( s_TerminatedSystemsPerBlock < SPB )
	{
		// INITIALISE TIME STEPPING -------------------------------------------
		Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
		for (int j=0; j<Launches; j++)
		{
			LocalSystemID = threadIdx.x + j*blockDim.x;
			
			if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
			{
				s_UpdateStep[LocalSystemID]           = 1;
				s_IsFinite[LocalSystemID]             = 1;
				s_EndTimeDomainReached[LocalSystemID] = 0;
				
				s_TimeStep[LocalSystemID] = s_NewTimeStep[LocalSystemID];
				
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
		if ( Algorithm == RK4 ) // Eliminated at compile if Algorithm != RK4
		{
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_Stepper_RK4<NumberOfBlockLaunches,NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>( \
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
			
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_ErrorController_RK4<NumberOfBlockLaunches, UPS, SPB, Precision>( \
				s_IsFinite, \
				s_NewTimeStep, \
				SolverOptions.InitialTimeStep, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep);
		}
		
		if ( Algorithm == RKCK45 ) // Eliminated at compile if Algorithm != RKCK45
		{
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_Stepper_RKCK45<NumberOfBlockLaunches,NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>( \
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
			
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_ErrorController_RKCK45<NumberOfBlockLaunches, UPS, UD, SPB, Precision>( \
				s_IsFinite, \
				s_TimeStep, \
				s_NewTimeStep, \
				r_ActualState, \
				r_NextState, \
				r_Error, \
				s_RelativeTolerance, \
				s_AbsoluteTolerance, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep, \
				SolverOptions);
		}
		
		
		// NEW EVENT VALUE AND TIME STEP CONTROL-------------------------------
		if ( NE > 0 ) // Eliminated at compile time if NE=0
		{
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
				LocalSystemID         = LocalThreadID_Logical / UPS;
				UnitID                = LocalThreadID_Logical % UPS;
				GlobalSystemID        = LocalSystemID + BlockID*SPB;
				
				if ( ( LocalSystemID < SPB ) && ( s_TerminateSystemScope[LocalSystemID] == 0 ) )
				{
					CoupledSystems_PerBlock_EventFunction<Precision>(\
						GlobalSystemID, \
						UnitID, \
						&r_NextEventValue[BL][0], \
						s_ActualTime[LocalSystemID]+s_TimeStep[LocalSystemID], \
						s_NewTimeStep[LocalSystemID], \
						&s_TimeDomain[LocalSystemID][0], \
						&r_NextState[BL][0], \
						&r_UnitParameters[BL][0], \
						&s_SystemParameters[LocalSystemID][0], \
						gs_GlobalParameters, \
						gs_IntegerGlobalParameters, \
						&r_UnitAccessories[BL][0], \
						&r_IntegerUnitAccessories[BL][0], \
						&s_SystemAccessories[LocalSystemID][0], \
						&s_IntegerSystemAccessories[LocalSystemID][0]);
				}
			}
			__syncthreads();
			
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_EventTimeStepControl<NumberOfBlockLaunches, UPS, SPB, NE, Precision>(\
				s_TimeStep, \
				s_NewTimeStep, \
				s_TerminateSystemScope, \
				s_UpdateStep, \
				r_ActualEventValue, \
				r_NextEventValue, \
				s_EventTolerance, \
				s_EventDirection, \
				SolverOptions.MinimumTimeStep);
		}
		
		
		// UPDATE PROCESS -----------------------------------------------------
		// Update actual state and time for an accepted step
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
			LocalSystemID         = LocalThreadID_Logical / UPS;
			UnitID                = LocalThreadID_Logical % UPS;
			
			if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
			{
				if ( UnitID == 0 )
					s_ActualTime[LocalSystemID] += s_TimeStep[LocalSystemID];
				
				for (int i=0; i<UD; i++)
					r_ActualState[BL][i] = r_NextState[BL][i];
			}
		}
		__syncthreads();
		
		// Call user defined ActionAfterSuccessfulTimeStep
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
			LocalSystemID         = LocalThreadID_Logical / UPS;
			UnitID                = LocalThreadID_Logical % UPS;
			GlobalSystemID        = LocalSystemID + BlockID*SPB;
			
			if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
			{
				CoupledSystems_PerBlock_ActionAfterSuccessfulTimeStep<Precision>(\
					GlobalSystemID, \
					UnitID, \
					s_UserDefinedTermination[LocalSystemID], \
					s_ActualTime[LocalSystemID], \
					s_TimeStep[LocalSystemID], \
					&s_TimeDomain[LocalSystemID][0], \
					&r_ActualState[BL][0], \
					&r_UnitParameters[BL][0], \
					&s_SystemParameters[LocalSystemID][0], \
					gs_GlobalParameters, \
					gs_IntegerGlobalParameters, \
					&r_UnitAccessories[BL][0], \
					&r_IntegerUnitAccessories[BL][0], \
					&s_SystemAccessories[LocalSystemID][0], \
					&s_IntegerSystemAccessories[LocalSystemID][0]);
			}
		}
		__syncthreads();
		
		// Event handling actions; Eliminated at compile time if NE=0
		if ( NE > 0 )
		{
			// Call user defined ActionAfterEventDetection if event is detected
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
				LocalSystemID         = LocalThreadID_Logical / UPS;
				UnitID                = LocalThreadID_Logical % UPS;
				GlobalSystemID        = LocalSystemID + BlockID*SPB;
				
				if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
				{	
					for (int i=0; i<NE; i++)
					{
						if ( ( ( r_ActualEventValue[BL][i] >  s_EventTolerance[i] ) && ( abs(r_NextEventValue[BL][i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
							 ( ( r_ActualEventValue[BL][i] < -s_EventTolerance[i] ) && ( abs(r_NextEventValue[BL][i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
						{
							CoupledSystems_PerBlock_ActionAfterEventDetection<Precision>(\
								GlobalSystemID, \
								UnitID, \
								i, \
								s_UserDefinedTermination[LocalSystemID], \
								s_ActualTime[LocalSystemID], \
								s_TimeStep[LocalSystemID], \
								&s_TimeDomain[LocalSystemID][0], \
								&r_ActualState[BL][0], \
								&r_UnitParameters[BL][0], \
								&s_SystemParameters[LocalSystemID][0], \
								gs_GlobalParameters, \
								gs_IntegerGlobalParameters, \
								&r_UnitAccessories[BL][0], \
								&r_IntegerUnitAccessories[BL][0], \
								&s_SystemAccessories[LocalSystemID][0], \
								&s_IntegerSystemAccessories[LocalSystemID][0]);
						}
					}
				}
			}
			__syncthreads();
			
			// Update next event value (modification of state variables is possuble by the user)
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
				LocalSystemID         = LocalThreadID_Logical / UPS;
				UnitID                = LocalThreadID_Logical % UPS;
				GlobalSystemID        = LocalSystemID + BlockID*SPB;
				
				if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
				{
					CoupledSystems_PerBlock_EventFunction<Precision>(\
						GlobalSystemID, \
						UnitID, \
						&r_NextEventValue[BL][0], \
						s_ActualTime[LocalSystemID], \
						s_NewTimeStep[LocalSystemID], \
						&s_TimeDomain[LocalSystemID][0], \
						&r_ActualState[BL][0], \
						&r_UnitParameters[BL][0], \
						&s_SystemParameters[LocalSystemID][0], \
						gs_GlobalParameters, \
						gs_IntegerGlobalParameters, \
						&r_UnitAccessories[BL][0], \
						&r_IntegerUnitAccessories[BL][0], \
						&s_SystemAccessories[LocalSystemID][0], \
						&s_IntegerSystemAccessories[LocalSystemID][0]);
				}
			}
			__syncthreads();	
				
			// Update event values
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
				LocalSystemID         = LocalThreadID_Logical / UPS;
				
				if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
				{
					for (int i=0; i<NE; i++)
						r_ActualEventValue[BL][i] = r_NextEventValue[BL][i];
				}
			}
			__syncthreads();
		}
		
		// Dense output actions; Eliminated at compile time if NDO=0
		if ( NDO > 0)
		{
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_DenseOutputStorageCondition<NumberOfBlockLaunches, UPS, SPB, NDO, Precision>(\
				s_EndTimeDomainReached, \
				s_UserDefinedTermination, \
				s_UpdateStep, \
				s_UpdateDenseOutput, \
				s_DenseOutputIndex, \
				s_NumberOfSkippedStores, \
				s_DenseOutputActualTime, \
				s_ActualTime, \
				SolverOptions);
			
			CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_StoreDenseOutput<NumberOfBlockLaunches, NS, UPS, UD, SPB, Precision>(\
				s_UpdateStep, \
				s_UpdateDenseOutput, \
				s_DenseOutputIndex, \
				s_NumberOfSkippedStores, \
				GlobalVariables.d_DenseOutputTimeInstances, \
				GlobalVariables.d_DenseOutputStates, \
				s_DenseOutputActualTime, \
				s_ActualTime, \
				r_ActualState, \
				s_TimeDomain, \
				ThreadConfiguration, \
				SolverOptions);
		}
		
		// Check termination
		Launches = SPB / blockDim.x + (SPB % blockDim.x == 0 ? 0 : 1);
		for (int j=0; j<Launches; j++)
		{
			LocalSystemID = threadIdx.x + j*blockDim.x;
			
			if ( ( LocalSystemID < SPB ) && ( s_UpdateStep[LocalSystemID] == 1 ) )
			{
				if ( ( s_EndTimeDomainReached[LocalSystemID] == 1 ) || ( s_UserDefinedTermination[LocalSystemID] == 1 ) )
				{
					s_TerminateSystemScope[LocalSystemID] = 1;
					atomicAdd(&s_TerminatedSystemsPerBlock, 1);
					
					s_UpdateStep[LocalSystemID] = 0;
				}
			}
		}
		__syncthreads();
		
	}
	__syncthreads();
	
	
	// FINALISATION
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_Finalization<Precision>(\
				GlobalSystemID, \
				UnitID, \
				s_ActualTime[LocalSystemID], \
				s_TimeStep[LocalSystemID], \
				&s_TimeDomain[LocalSystemID][0], \
				&r_ActualState[BL][0], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0]);
		}
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
		LocalSystemID  = threadIdx.x   + j*blockDim.x;
		GlobalSystemID = LocalSystemID + BlockID*SPB;
		
		if ( ( LocalSystemID < SPB ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
		{
			GlobalVariables.d_ActualTime[GlobalSystemID] = s_ActualTime[LocalSystemID];
			
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


template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
__global__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables<Precision> GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions<Precision> SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LogicalThreadsPerBlock      = SPB * UPS;
	const int NumberOfBlockLaunches       = LogicalThreadsPerBlock / TPB + (LogicalThreadsPerBlock % TPB == 0 ? 0 : 1);
	const int ThreadPaddingPerBlock       = NumberOfBlockLaunches * TPB - LogicalThreadsPerBlock;
	const int TotalLogicalThreads         = (LogicalThreadsPerBlock + ThreadPaddingPerBlock) * gridDim.x;
	const int TotalLogicalThreadsPerBlock = (LogicalThreadsPerBlock + ThreadPaddingPerBlock);
	const int LocalThreadID_GPU           = threadIdx.x;
	const int GlobalSystemID              = blockIdx.x;
	
	int GlobalThreadID_Logical; // Depends on BlockLaunch
	int GlobalMemoryID;         // Depends on BlockLaunch
	int UnitID;                 // Depends on BlockLaunch
	
	
	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    SYSTEMS ARE PADDED IN SHARED MEMORY (NO GLOBAL BOUND CHECK IS NECESSARY AFTER LOADING)
	//    MINIMUM ALLOCABLE MEMORY IS 1
	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;
	
	Precision* gs_GlobalParameters = (Precision*)&DynamicSharedMemory;
		MemoryShift = (SharedMemoryUsage.GlobalVariables  == 1 ? NGP : 0);
	
	Precision* gs_CouplingMatrix = (Precision*)&gs_GlobalParameters[MemoryShift];
		MemoryShift = (SharedMemoryUsage.CouplingMatrices == 1 ? NC*SharedMemoryUsage.SingleCouplingMatrixSize : 0);
	
	int* gs_IntegerGlobalParameters = (int*)&gs_CouplingMatrix[MemoryShift];
	
	const bool IsAdaptive  = ( Algorithm==RK4 ? 0 : 1 );
	
	// Bank conflict if NCmod = 0, 2, 4, 8 or 16
	const int NCmod        = NC % 32;
	const int IsPowerOfTwo = !( NCmod & (NCmod-1) ); // Including 0!
	const int NCp          = NC + ( NCmod==1 ? 0 : ( IsPowerOfTwo==1 ? 1 : 0 ) );
	// Bank conflicts if NSP, NSA and NiSA = 4, 8 or 16
	const int NSPp  = (  NSP==4 ?  NSP+1 : (  NSP==8 ?  NSP+1 : (  NSP==16 ?  NSP+1 : NSP  ) ) );
	const int NSAp  = (  NSA==4 ?  NSA+1 : (  NSA==8 ?  NSA+1 : (  NSA==16 ?  NSA+1 : NSA  ) ) );
	const int NiSAp = ( NiSA==4 ? NiSA+1 : ( NiSA==8 ? NiSA+1 : ( NiSA==16 ? NiSA+1 : NiSA ) ) );
	
	__shared__ Precision s_CouplingTerms[UPS][NCp];                       // Need access by user
	__shared__ Precision s_CouplingStrength[NCp];                         // Internal
	__shared__ Precision s_TimeDomain[2];                                 // Need access by user
	__shared__ Precision s_ActualTime;                                    // Need access by user
	__shared__ Precision s_TimeStep;                                      // Need access by user
	__shared__ Precision s_NewTimeStep;                                   // Need access by user
	__shared__ Precision s_SystemParameters[ (NSPp==0 ? 1 : NSPp) ];      // Need access by user
	__shared__ Precision s_SystemAccessories[ (NSAp==0 ? 1 : NSAp) ];     // Need access by user
	__shared__ Precision s_RelativeTolerance[ (IsAdaptive==0 ? 1 : UD) ]; // Internal
	__shared__ Precision s_AbsoluteTolerance[ (IsAdaptive==0 ? 1 : UD) ]; // Internal
	__shared__ Precision s_EventTolerance[ (NE==0 ? 1 : NE) ];            // Internal
	__shared__ Precision s_DenseOutputActualTime;                         // Internal
	__shared__ int s_CouplingIndex[NC];                                   // Internal
	__shared__ int s_DenseOutputIndex;                                    // Internal
	__shared__ int s_UpdateDenseOutput;                                   // Internal
	__shared__ int s_NumberOfSkippedStores;                               // Internal
	__shared__ int s_IntegerSystemAccessories[ (NiSAp==0 ? 1 : NiSAp) ];  // Need access by user
	__shared__ int s_EventDirection[ (NE==0 ? 1 : NE) ];                  // Need access by user
	__shared__ int s_TerminatedSystemsPerBlock;                           // Internal
	__shared__ int s_IsFinite;                                            // Internal
	__shared__ int s_TerminateSystemScope;                                // Internal
	__shared__ int s_UserDefinedTermination;                              // Internal
	__shared__ int s_UpdateStep;                                          // Internal
	__shared__ int s_EndTimeDomainReached;                                // Internal
	
	// Initialise block scope variables
	if ( LocalThreadID_GPU == 0 )
		s_TerminatedSystemsPerBlock = 0;
	__syncthreads();
	
	// Initialise system scope variables
	if ( ( threadIdx.x == 0 ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
	{
		for (int i=0; i<2; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_TimeDomain[i] = GlobalVariables.d_TimeDomain[GlobalMemoryID];
			
			if ( i==0 )
			{
				s_ActualTime  = s_TimeDomain[i];
				s_TimeStep    = SolverOptions.InitialTimeStep;
				s_NewTimeStep = SolverOptions.InitialTimeStep;
			}
		}
		
		for (int i=0; i<NSP; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_SystemParameters[i] = GlobalVariables.d_SystemParameters[GlobalMemoryID];
		}
		
		for (int i=0; i<NSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_SystemAccessories[i] = GlobalVariables.d_SystemAccessories[GlobalMemoryID];
		}
		
		for (int i=0; i<NiSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_IntegerSystemAccessories[i] = GlobalVariables.d_IntegerSystemAccessories[GlobalMemoryID];
		}
		
		for (int i=0; i<NC; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_CouplingStrength[i] = GlobalVariables.d_CouplingStrength[GlobalMemoryID];
		}
		
		for (int i=0; i<1; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_DenseOutputIndex       = GlobalVariables.d_DenseOutputIndex[GlobalMemoryID];
			s_DenseOutputActualTime  = s_TimeDomain[0];
			s_UpdateDenseOutput      = 1;
			s_NumberOfSkippedStores  = 0;
			s_TerminateSystemScope   = 0;
			s_UserDefinedTermination = 0;
		}
	}
	
	if ( ( threadIdx.x == 0 ) && ( ( GlobalSystemID >= SolverOptions.ActiveSystems ) || ( GlobalSystemID >= NS ) ) )
	{
		atomicAdd(&s_TerminatedSystemsPerBlock, 1);
		s_TerminateSystemScope = 1;
		s_UpdateStep           = 0;
		s_UpdateDenseOutput    = 0;
	}
	
	// Initialise solver tolerances
	if ( IsAdaptive  == 1 )
	{
		int Launches = UD / blockDim.x + (UD % blockDim.x == 0 ? 0 : 1);
		
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
	int Launches = NE / blockDim.x + (NE % blockDim.x == 0 ? 0 : 1);
	for (int i=0; i<Launches; i++)
	{
		int idx = threadIdx.x + i*blockDim.x;
		
		if ( idx < NE )
		{
			s_EventTolerance[idx]   = GlobalVariables.d_EventTolerance[idx];
			s_EventDirection[idx]   = GlobalVariables.d_EventDirection[idx];
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
		int MaxElementNumber = NC*SharedMemoryUsage.SingleCouplingMatrixSize;
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
	Precision r_ActualEventValue[ (NE==0 ? 1 : NumberOfBlockLaunches) ][ (NE==0 ? 1 : NE) ];
	Precision r_NextEventValue[ (NE==0 ? 1 : NumberOfBlockLaunches) ][ (NE==0 ? 1 : NE) ];
	Precision r_Error[NumberOfBlockLaunches][UD];
	Precision r_UnitParameters[ (NUP==0 ? 1 : NumberOfBlockLaunches) ][ (NUP==0 ? 1 : NUP) ];
	Precision r_UnitAccessories[ (NUA==0 ? 1 : NumberOfBlockLaunches) ][ (NUA==0 ? 1 : NUA) ];
	int       r_IntegerUnitAccessories[ (NiUA==0 ? 1 : NumberOfBlockLaunches) ][ (NiUA==0 ? 1 : NiUA) ];
	
	// Initialise unit scope variables
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		GlobalThreadID_Logical = UnitID + GlobalSystemID*TotalLogicalThreadsPerBlock;
		
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
	
	
	// INITIALISATION
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( ( UnitID < UPS ) && ( s_TerminateSystemScope == 0 ) )
		{
			CoupledSystems_PerBlock_Initialization<Precision>(\
				GlobalSystemID, \
				UnitID, \
				s_ActualTime, \
				s_TimeStep, \
				&s_TimeDomain[0], \
				&r_ActualState[BL][0], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0]);
		}
	}
	__syncthreads();
	
	if ( NE > 0 ) // Eliminated at compile time if NE=0
	{
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			UnitID = LocalThreadID_GPU + BL*blockDim.x;
			
			if ( ( UnitID < UPS ) && ( s_TerminateSystemScope == 0 ) )
			{
				CoupledSystems_PerBlock_EventFunction<Precision>(\
					GlobalSystemID, \
					UnitID, \
					&r_ActualEventValue[BL][0], \
					s_ActualTime, \
					s_TimeStep, \
					&s_TimeDomain[0], \
					&r_ActualState[BL][0], \
					&r_UnitParameters[BL][0], \
					&s_SystemParameters[0], \
					gs_GlobalParameters, \
					gs_IntegerGlobalParameters, \
					&r_UnitAccessories[BL][0], \
					&r_IntegerUnitAccessories[BL][0], \
					&s_SystemAccessories[0], \
					&s_IntegerSystemAccessories[0]);
			}
		}
		__syncthreads();
	}
	
	if ( NDO > 0 ) // Eliminated at compile time if NDO=0
	{
		CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_StoreDenseOutput<NumberOfBlockLaunches, NS, UPS, UD, Precision>(\
			s_UpdateStep, \
			s_UpdateDenseOutput, \
			s_DenseOutputIndex, \
			s_NumberOfSkippedStores, \
			GlobalVariables.d_DenseOutputTimeInstances, \
			GlobalVariables.d_DenseOutputStates, \
			s_DenseOutputActualTime, \
			s_ActualTime, \
			r_ActualState, \
			s_TimeDomain, \
			ThreadConfiguration, \
			SolverOptions);
	}
	
	
	// SOLVER MANAGEMENT ------------------------------------------------------
	while ( s_TerminatedSystemsPerBlock < SPB )
	{
		// INITIALISE TIME STEPPING -------------------------------------------
		if ( ( threadIdx.x == 0 ) && ( s_TerminateSystemScope == 0 ) )
		{
			s_UpdateStep           = 1;
			s_IsFinite             = 1;
			s_EndTimeDomainReached = 0;
			
			s_TimeStep = s_NewTimeStep;
			
			if ( s_TimeStep > ( s_TimeDomain[1] - s_ActualTime ) )
			{
				s_TimeStep = s_TimeDomain[1] - s_ActualTime;
				s_TimeStep = MPGOS::FMAX(s_TimeStep, SolverOptions.MinimumTimeStep);
				
				s_EndTimeDomainReached = 1;
			}
		}
		__syncthreads();
		
		
		// STEPPER ------------------------------------------------------------
		if ( Algorithm == RK4 ) // Eliminated at compile if Algorithm != RK4
		{
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_Stepper_RK4<NumberOfBlockLaunches,NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
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
			
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_ErrorController_RK4<Precision>(\
				s_IsFinite, \
				s_NewTimeStep, \
				SolverOptions.InitialTimeStep, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep);
		}
		
		if ( Algorithm == RKCK45 ) // Eliminated at compile if Algorithm != RKCK45
		{
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_Stepper_RKCK45<NumberOfBlockLaunches,NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
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
			
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_ErrorController_RKCK45<NumberOfBlockLaunches, UPS, UD, Precision>(\
				s_IsFinite, \
				s_TimeStep, \
				s_NewTimeStep, \
				r_ActualState, \
				r_NextState, \
				r_Error, \
				s_RelativeTolerance, \
				s_AbsoluteTolerance, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep, \
				SolverOptions);
		}
		
		
		// NEW EVENT VALUE AND TIME STEP CONTROL-------------------------------
		if ( NE > 0 ) // Eliminated at compile time if NE=0
		{
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				UnitID = LocalThreadID_GPU + BL*blockDim.x;
				
				if ( ( UnitID < UPS ) && ( s_TerminateSystemScope == 0 ) )
				{
					CoupledSystems_PerBlock_EventFunction<Precision>(\
						GlobalSystemID, \
						UnitID, \
						&r_NextEventValue[BL][0], \
						s_ActualTime+s_TimeStep, \
						s_NewTimeStep, \
						&s_TimeDomain[0], \
						&r_NextState[BL][0], \
						&r_UnitParameters[BL][0], \
						&s_SystemParameters[0], \
						gs_GlobalParameters, \
						gs_IntegerGlobalParameters, \
						&r_UnitAccessories[BL][0], \
						&r_IntegerUnitAccessories[BL][0], \
						&s_SystemAccessories[0], \
						&s_IntegerSystemAccessories[0]);
				}
			}
			__syncthreads();
			
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_EventTimeStepControl<NumberOfBlockLaunches, UPS, NE, Precision>(\
				s_TimeStep, \
				s_NewTimeStep, \
				s_TerminateSystemScope, \
				s_UpdateStep, \
				r_ActualEventValue, \
				r_NextEventValue, \
				s_EventTolerance, \
				s_EventDirection, \
				SolverOptions.MinimumTimeStep);
		}
		
		
		// UPDATE PROCESS -----------------------------------------------------
		// Update actual state and time for an accepted step
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			UnitID = LocalThreadID_GPU + BL*blockDim.x;
			
			if ( ( UnitID < UPS ) && ( s_UpdateStep == 1 ) )
			{
				if ( UnitID == 0 )
					s_ActualTime += s_TimeStep;
				
				for (int i=0; i<UD; i++)
					r_ActualState[BL][i] = r_NextState[BL][i];
			}
		}
		__syncthreads();
		
		// Call user defined ActionAfterSuccessfulTimeStep
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
		{
			UnitID = LocalThreadID_GPU + BL*blockDim.x;
			
			if ( ( UnitID < UPS ) && ( s_UpdateStep == 1 ) )
			{
				CoupledSystems_PerBlock_ActionAfterSuccessfulTimeStep<Precision>(\
					GlobalSystemID, \
					UnitID, \
					s_UserDefinedTermination, \
					s_ActualTime, \
					s_TimeStep, \
					&s_TimeDomain[0], \
					&r_ActualState[BL][0], \
					&r_UnitParameters[BL][0], \
					&s_SystemParameters[0], \
					gs_GlobalParameters, \
					gs_IntegerGlobalParameters, \
					&r_UnitAccessories[BL][0], \
					&r_IntegerUnitAccessories[BL][0], \
					&s_SystemAccessories[0], \
					&s_IntegerSystemAccessories[0]);
			}
		}
		__syncthreads();
		
		// Event handling actions; Eliminated at compile time if NE=0
		if ( NE > 0 )
		{
			// Call user defined ActionAfterEventDetection if event is detected
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				UnitID = LocalThreadID_GPU + BL*blockDim.x;
				
				if ( ( UnitID < UPS ) && ( s_UpdateStep == 1 ) )
				{	
					for (int i=0; i<NE; i++)
					{
						if ( ( ( r_ActualEventValue[BL][i] >  s_EventTolerance[i] ) && ( abs(r_NextEventValue[BL][i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
							 ( ( r_ActualEventValue[BL][i] < -s_EventTolerance[i] ) && ( abs(r_NextEventValue[BL][i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
						{
							CoupledSystems_PerBlock_ActionAfterEventDetection<Precision>(\
								GlobalSystemID, \
								UnitID, \
								i, \
								s_UserDefinedTermination, \
								s_ActualTime, \
								s_TimeStep, \
								&s_TimeDomain[0], \
								&r_ActualState[BL][0], \
								&r_UnitParameters[BL][0], \
								&s_SystemParameters[0], \
								gs_GlobalParameters, \
								gs_IntegerGlobalParameters, \
								&r_UnitAccessories[BL][0], \
								&r_IntegerUnitAccessories[BL][0], \
								&s_SystemAccessories[0], \
								&s_IntegerSystemAccessories[0]);
						}
					}
				}
			}
			__syncthreads();
			
			// Update next event value (modification of state variables is possuble by the user)
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				UnitID = LocalThreadID_GPU + BL*blockDim.x;
				
				if ( ( UnitID < UPS ) && ( s_UpdateStep == 1 ) )
				{
					CoupledSystems_PerBlock_EventFunction<Precision>(\
						GlobalSystemID, \
						UnitID, \
						&r_NextEventValue[BL][0], \
						s_ActualTime, \
						s_NewTimeStep, \
						&s_TimeDomain[0], \
						&r_ActualState[BL][0], \
						&r_UnitParameters[BL][0], \
						&s_SystemParameters[0], \
						gs_GlobalParameters, \
						gs_IntegerGlobalParameters, \
						&r_UnitAccessories[BL][0], \
						&r_IntegerUnitAccessories[BL][0], \
						&s_SystemAccessories[0], \
						&s_IntegerSystemAccessories[0]);
				}
			}
			__syncthreads();	
				
			// Update event values
			for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			{
				UnitID = LocalThreadID_GPU + BL*blockDim.x;
				
				if ( ( UnitID < UPS ) && ( s_UpdateStep == 1 ) )
				{
					for (int i=0; i<NE; i++)
						r_ActualEventValue[BL][i] = r_NextEventValue[BL][i];
				}
			}
			__syncthreads();
		}
		
		// Dense output actions; Eliminated at compile time if NDO=0
		if ( NDO > 0)
		{
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_DenseOutputStorageCondition<NDO, Precision>(\
				s_EndTimeDomainReached, \
				s_UserDefinedTermination, \
				s_UpdateStep, \
				s_UpdateDenseOutput, \
				s_DenseOutputIndex, \
				s_NumberOfSkippedStores, \
				s_DenseOutputActualTime, \
				s_ActualTime, \
				SolverOptions);
			
			CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_StoreDenseOutput<NumberOfBlockLaunches, NS, UPS, UD, Precision>(\
				s_UpdateStep, \
				s_UpdateDenseOutput, \
				s_DenseOutputIndex, \
				s_NumberOfSkippedStores, \
				GlobalVariables.d_DenseOutputTimeInstances, \
				GlobalVariables.d_DenseOutputStates, \
				s_DenseOutputActualTime, \
				s_ActualTime, \
				r_ActualState, \
				s_TimeDomain, \
				ThreadConfiguration, \
				SolverOptions);
		}
		
		// Chect termination
		if ( ( threadIdx.x == 0 ) && ( s_UpdateStep == 1 ) )
		{
			if ( ( s_EndTimeDomainReached == 1 ) || ( s_UserDefinedTermination == 1 ) )
			{
				s_TerminateSystemScope = 1;
				atomicAdd(&s_TerminatedSystemsPerBlock, 1);
				
				s_UpdateStep = 0;
			}
		}
		__syncthreads();
		
	}
	__syncthreads();
	
	
	// FINALISATION
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_Finalization<Precision>(\
				GlobalSystemID, \
				UnitID, \
				s_ActualTime, \
				s_TimeStep, \
				&s_TimeDomain[0], \
				&r_ActualState[BL][0], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0]);
		}
	}
	__syncthreads();
	
	
	// WRITE DATA BACK TO GLOBAL MEMORY ---------------------------------------
	// Unit scope variables (register variables)
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		GlobalThreadID_Logical = UnitID + GlobalSystemID*TotalLogicalThreadsPerBlock;
		
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
	if ( ( threadIdx.x == 0 ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
	{
		GlobalVariables.d_ActualTime[GlobalSystemID] = s_ActualTime;
		
		for (int i=0; i<2; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_TimeDomain[GlobalMemoryID] = s_TimeDomain[i];
		}
		
		for (int i=0; i<NSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_SystemAccessories[GlobalMemoryID] = s_SystemAccessories[i];
		}
		
		for (int i=0; i<NiSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_IntegerSystemAccessories[GlobalMemoryID] = s_IntegerSystemAccessories[i];
		}
		
		for (int i=0; i<1; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_DenseOutputIndex[GlobalMemoryID] = s_DenseOutputIndex;
		}
	}
	
	__syncthreads();
}

#endif
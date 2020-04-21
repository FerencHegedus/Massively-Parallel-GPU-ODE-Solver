#ifndef COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_STEPPERS_H
#define COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_STEPPERS_H


// ----------------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_RK4( \
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision* s_ActualTime, \
			Precision* s_TimeStep, \
			int*       s_IsFinite, \
			Precision  r_UnitParameters[(NUP==0?1:NBL)][(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSP==0?1:SPB)][(NSP==0?1:NSP)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NBL)][(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NBL)][(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSA==0?1:SPB)][(NSA==0?1:NSA)], \
			int        s_IntegerSystemAccessories[(NiSA==0?1:SPB)][(NiSA==0?1:NiSA)], \
			Precision  s_CouplingTerms[SPB][UPS][NC], \
			Precision  r_CouplingFactor[NBL][NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[SPB][NC], \
			int        s_CouplingIndex[NC])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int LocalThreadID_GPU = threadIdx.x;
	int LocalThreadID_Logical;
	int LocalSystemID;
	int UnitID;
	
	
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time[SPB];
	
	Precision r_State[NBL][UD];
	
	Precision r_Stage1[NBL][UD];
	
	
	// K1 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_NextState[BL][0], &r_ActualState[BL][0], s_ActualTime[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_NextState[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + r_NextState[BL][i] * 0.5*s_TimeStep[LocalSystemID];
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + 0.5*s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage1[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] += 2*r_Stage1[BL][i];
				r_State[BL][i] = r_ActualState[BL][i] + r_Stage1[BL][i]*0.5*s_TimeStep[LocalSystemID];
			}
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage1[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] += 2*r_Stage1[BL][i];
				r_State[BL][i] = r_ActualState[BL][i] + r_Stage1[BL][i]*s_TimeStep[LocalSystemID];
			}
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage1[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// NEW STATE --------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Finalize new state
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] = r_ActualState[BL][i] + s_TimeStep[LocalSystemID]*( r_NextState[BL][i] + r_Stage1[BL][i] )*(1.0/6.0);
				
				if ( isfinite( r_NextState[BL][i] ) == 0 )
					s_IsFinite[LocalSystemID] = 0;
			}
		}
	}
	__syncthreads();
}


// ----------------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_RKCK45( \
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision* s_ActualTime, \
			Precision* s_TimeStep, \
			int*       s_IsFinite, \
			Precision  r_Error[NBL][UD], \
			Precision  r_UnitParameters[(NUP==0?1:NBL)][(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSP==0?1:SPB)][(NSP==0?1:NSP)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NBL)][(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NBL)][(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSA==0?1:SPB)][(NSA==0?1:NSA)], \
			int        s_IntegerSystemAccessories[(NiSA==0?1:SPB)][(NiSA==0?1:NiSA)], \
			Precision  s_CouplingTerms[SPB][UPS][NC], \
			Precision  r_CouplingFactor[NBL][NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[SPB][NC], \
			int        s_CouplingIndex[NC])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int LocalThreadID_GPU = threadIdx.x;
	int LocalThreadID_Logical;
	int LocalSystemID;
	int UnitID;
	
	
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time[SPB];
	
	Precision r_State[NBL][UD];
	
	Precision r_Stage1[NBL][UD];
	Precision r_Stage2[NBL][UD];
	Precision r_Stage3[NBL][UD];
	Precision r_Stage4[NBL][UD];
	Precision r_Stage5[NBL][UD];
	Precision r_Stage6[NBL][UD];
	
	
	// K1 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage1[BL][0], &r_ActualState[BL][0], s_ActualTime[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(1.0/5.0) ) * s_TimeStep[LocalSystemID];
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(1.0/5.0)*s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage2[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage2[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(3.0/40.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(9.0/40.0) ) * s_TimeStep[LocalSystemID];
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(3.0/10.0)*s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage3[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage3[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(3.0/10.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(-9.0/10.0) + \
														  r_Stage3[BL][i] * static_cast<Precision>(6.0/5.0) ) * s_TimeStep[LocalSystemID];
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(3.0/5.0)*s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage4[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage4[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K5 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(-11.0/54.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(5.0/2.0) + \
														  r_Stage3[BL][i] * static_cast<Precision>(-70.0/27.0) + \
														  r_Stage4[BL][i] * static_cast<Precision>(35.0/27.0) ) * s_TimeStep[LocalSystemID];
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage5[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage5[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K6 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(1631.0/55296.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(175.0/512.0) + \
														  r_Stage3[BL][i] * static_cast<Precision>(575.0/13824.0) + \
														  r_Stage4[BL][i] * static_cast<Precision>(44275.0/110592.0) + \
														  r_Stage5[BL][i] * static_cast<Precision>(253.0/4096.0) ) * s_TimeStep[LocalSystemID];
			
			if ( UnitID == 0 )
				s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(7.0/8.0)*s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(&r_Stage6[BL][0], &r_State[BL][0], s_Time[LocalSystemID], \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[LocalSystemID][0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[LocalSystemID][0], \
				&s_IntegerSystemAccessories[LocalSystemID][0], \
				&s_CouplingTerms[LocalSystemID][UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = 0;
				int MemoryShift = i*UPS*UPS;
				int Row = UnitID;
				int idx;
				
				for (int Col=0; Col<UPS; Col++)
				{
					idx = Row + Col*UPS + MemoryShift;
					CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][i];
				}
		
				r_Stage6[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// NEW STATE AND ERROR ----------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Finalize new state
	{
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		
		if ( LocalSystemID < SPB )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(37.0/378.0) + \
                                                              r_Stage3[BL][i] * static_cast<Precision>(250.0/621.0) + \
														      r_Stage4[BL][i] * static_cast<Precision>(125.0/594.0) + \
														      r_Stage6[BL][i] * static_cast<Precision>(512.0/1771.0) ) * s_TimeStep[LocalSystemID];
				
				r_Error[BL][i] = r_Stage1[BL][i] * ( static_cast<Precision>(  37.0/378.0  -  2825.0/27648.0 ) ) + \
                                 r_Stage3[BL][i] * ( static_cast<Precision>( 250.0/621.0  - 18575.0/48384.0 ) ) + \
								 r_Stage4[BL][i] * ( static_cast<Precision>( 125.0/594.0  - 13525.0/55296.0 ) ) + \
								 r_Stage5[BL][i] * ( static_cast<Precision>(   0.0        -   277.0/14336.0 ) ) + \
								 r_Stage6[BL][i] * ( static_cast<Precision>( 512.0/1771.0 -     1.0/4.0 ) );
				r_Error[BL][i] = s_TimeStep[LocalSystemID]*abs( r_Error[BL][i] ) + 1e-18;
				
				if ( ( isfinite( r_NextState[BL][i] ) == 0 ) || ( isfinite( r_Error[BL][i] ) == 0 ) )
					s_IsFinite[LocalSystemID] = 0;
			}
		}
	}
	__syncthreads();
}


#endif
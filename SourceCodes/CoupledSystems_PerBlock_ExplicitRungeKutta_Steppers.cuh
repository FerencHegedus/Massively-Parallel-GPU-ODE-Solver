#ifndef COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_STEPPERS_H
#define COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_STEPPERS_H

template <int UPS, int SPB, int NC, int NCp, int CBW, int CCI, class Precision>
__forceinline__ __device__ Precision ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches(\
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingTerms[SPB][UPS][NCp], \
			int        LocalSystemID, \
			int        UnitID, \
			int        CouplingSerialNumber);

template <int UPS, int NC, int NCp, int CBW, int CCI, class Precision>
__forceinline__ __device__ Precision ComputeCouplingValue_SingleSystem_MultipleBlockLaunches(\
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingTerms[UPS][NCp], \
			int        UnitID, \
			int        CouplingSerialNumber);


// MSMBL ----------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_Stepper_RK4(\
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision* s_ActualTime, \
			Precision* s_TimeStep, \
			int*       s_IsFinite, \
			Precision  r_UnitParameters[(NUP==0?1:NBL)][(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:SPB)][(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NBL)][(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NBL)][(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:SPB)][(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:SPB)][(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[SPB][UPS][NCp], \
			Precision  r_CouplingFactor[NBL][NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[SPB][NCp], \
			int        s_CouplingIndex[NCp])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LocalThreadID_GPU  = threadIdx.x;
	const int BlockID            = blockIdx.x;
	
	int LocalThreadID_Logical;
	int GlobalSystemID;
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_NextState[BL][0], \
				&r_ActualState[BL][0], \
				s_ActualTime[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_Stepper_RKCK45(\
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision* s_ActualTime, \
			Precision* s_TimeStep, \
			int*       s_IsFinite, \
			Precision  r_Error[NBL][UD], \
			Precision  r_UnitParameters[(NUP==0?1:NBL)][(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:SPB)][(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NBL)][(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NBL)][(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:SPB)][(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:SPB)][(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[SPB][UPS][NCp], \
			Precision  r_CouplingFactor[NBL][NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[SPB][NCp], \
			int        s_CouplingIndex[NCp])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LocalThreadID_GPU  = threadIdx.x;
	const int BlockID            = blockIdx.x;
	
	int LocalThreadID_Logical;
	int GlobalSystemID;
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_ActualState[BL][0], \
				s_ActualTime[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage2[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage3[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage4[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage5[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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
		GlobalSystemID        = LocalSystemID + BlockID*SPB;
		
		if ( LocalSystemID < SPB )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage6[BL][0], \
				&r_State[BL][0], \
				s_Time[LocalSystemID], \
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
				Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
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


// SSMBL ----------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_Stepper_RK4( \
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision  s_ActualTime, \
			Precision  s_TimeStep, \
			int&       s_IsFinite, \
			Precision  r_UnitParameters[(NUP==0?1:NBL)][(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NBL)][(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NBL)][(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[UPS][NCp], \
			Precision  r_CouplingFactor[NBL][NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[NCp], \
			int        s_CouplingIndex[NCp])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LocalThreadID_GPU = threadIdx.x;
	const int GlobalSystemID    = blockIdx.x;
	
	int UnitID;
	
	
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time;
	
	Precision r_State[NBL][UD];
	Precision r_Stage1[NBL][UD];
	
	
	// K1 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_NextState[BL][0], \
				&r_ActualState[BL][0], \
				s_ActualTime, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_NextState[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + r_NextState[BL][i] * 0.5*s_TimeStep;
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + 0.5*s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] += 2*r_Stage1[BL][i];
				r_State[BL][i] = r_ActualState[BL][i] + r_Stage1[BL][i]*0.5*s_TimeStep;
			}
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] += 2*r_Stage1[BL][i];
				r_State[BL][i] = r_ActualState[BL][i] + r_Stage1[BL][i]*s_TimeStep;
			}
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// NEW STATE --------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Finalize new state
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] = r_ActualState[BL][i] + s_TimeStep*( r_NextState[BL][i] + r_Stage1[BL][i] )*(1.0/6.0);
				
				if ( isfinite( r_NextState[BL][i] ) == 0 )
					s_IsFinite = 0;
			}
		}
	}
	__syncthreads();
}


// ----------------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches_Stepper_RKCK45(\
			Precision  r_ActualState[NBL][UD], \
			Precision  r_NextState[NBL][UD], \
			Precision  s_ActualTime, \
			Precision  s_TimeStep, \
			int&       s_IsFinite, \
			Precision  r_Error[NBL][UD], \
			Precision  r_UnitParameters[(NUP==0?1:NBL)][(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NBL)][(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NBL)][(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[UPS][NCp], \
			Precision  r_CouplingFactor[NBL][NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[NCp], \
			int        s_CouplingIndex[NCp])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LocalThreadID_GPU = threadIdx.x;
	const int GlobalSystemID    = blockIdx.x;
	
	int UnitID;
	
	
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time;
	
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
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage1[BL][0], \
				&r_ActualState[BL][0], \
				s_ActualTime, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage1[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(1.0/5.0) ) * s_TimeStep;
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + static_cast<Precision>(1.0/5.0)*s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage2[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage2[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(3.0/40.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(9.0/40.0) ) * s_TimeStep;
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + static_cast<Precision>(3.0/10.0)*s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage3[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage3[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(3.0/10.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(-9.0/10.0) + \
														  r_Stage3[BL][i] * static_cast<Precision>(6.0/5.0) ) * s_TimeStep;
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + static_cast<Precision>(3.0/5.0)*s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage4[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage4[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K5 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(-11.0/54.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(5.0/2.0) + \
														  r_Stage3[BL][i] * static_cast<Precision>(-70.0/27.0) + \
														  r_Stage4[BL][i] * static_cast<Precision>(35.0/27.0) ) * s_TimeStep;
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage5[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage5[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// K6 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // New location of function evaluation
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
				r_State[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(1631.0/55296.0) + \
                                                          r_Stage2[BL][i] * static_cast<Precision>(175.0/512.0) + \
														  r_Stage3[BL][i] * static_cast<Precision>(575.0/13824.0) + \
														  r_Stage4[BL][i] * static_cast<Precision>(44275.0/110592.0) + \
														  r_Stage5[BL][i] * static_cast<Precision>(253.0/4096.0) ) * s_TimeStep;
			
			if ( UnitID == 0 )
				s_Time = s_ActualTime + static_cast<Precision>(7.0/8.0)*s_TimeStep;
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Uncoupled right-hand side, coupling term and coupling factor
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			CoupledSystems_PerBlock_OdeFunction<Precision>(\
				GlobalSystemID, \
				UnitID, \
				&r_Stage6[BL][0], \
				&r_State[BL][0], \
				s_Time, \
				&r_UnitParameters[BL][0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[BL][0], \
				&r_IntegerUnitAccessories[BL][0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0], \
				&s_CouplingTerms[UnitID][0], \
				&r_CouplingFactor[BL][0]);
		}
	}
	__syncthreads();
	
	for (int BL=0; BL<NBL; BL++) // Extension with coupling
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<NC; i++)
			{
				Precision CouplingValue = ComputeCouplingValue_SingleSystem_MultipleBlockLaunches<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, UnitID, i);
				r_Stage6[BL][ s_CouplingIndex[i] ] += s_CouplingStrength[i]*r_CouplingFactor[BL][i]*CouplingValue;
			}
		}
	}
	__syncthreads();
	
	
	// NEW STATE AND ERROR ----------------------------------------------------
	for (int BL=0; BL<NBL; BL++) // Finalize new state
	{
		UnitID = LocalThreadID_GPU + BL*blockDim.x;
		
		if ( UnitID < UPS )
		{
			for (int i=0; i<UD; i++)
			{
				r_NextState[BL][i] = r_ActualState[BL][i] + ( r_Stage1[BL][i] * static_cast<Precision>(37.0/378.0) + \
                                                              r_Stage3[BL][i] * static_cast<Precision>(250.0/621.0) + \
														      r_Stage4[BL][i] * static_cast<Precision>(125.0/594.0) + \
														      r_Stage6[BL][i] * static_cast<Precision>(512.0/1771.0) ) * s_TimeStep;
				
				r_Error[BL][i] = r_Stage1[BL][i] * ( static_cast<Precision>(  37.0/378.0  -  2825.0/27648.0 ) ) + \
                                 r_Stage3[BL][i] * ( static_cast<Precision>( 250.0/621.0  - 18575.0/48384.0 ) ) + \
								 r_Stage4[BL][i] * ( static_cast<Precision>( 125.0/594.0  - 13525.0/55296.0 ) ) + \
								 r_Stage5[BL][i] * ( static_cast<Precision>(   0.0        -   277.0/14336.0 ) ) + \
								 r_Stage6[BL][i] * ( static_cast<Precision>( 512.0/1771.0 -     1.0/4.0 ) );
				r_Error[BL][i] = s_TimeStep*abs( r_Error[BL][i] ) + 1e-18;
				
				if ( ( isfinite( r_NextState[BL][i] ) == 0 ) || ( isfinite( r_Error[BL][i] ) == 0 ) )
					s_IsFinite = 0;
			}
		}
	}
	__syncthreads();
}


// MSSBL ----------------------------------------------------------------------
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_Stepper_RK4(\
			const int  LocalSystemID, \
			const int  UnitID, \
			const int  GlobalSystemID, \
			Precision  r_ActualState[UD], \
			Precision  r_NextState[UD], \
			Precision* s_ActualTime, \
			Precision* s_TimeStep, \
			int*       s_IsFinite, \
			Precision  r_UnitParameters[(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:SPB)][(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:SPB)][(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:SPB)][(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[SPB][UPS][NCp], \
			Precision  r_CouplingFactor[NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[SPB][NCp], \
			int        s_CouplingIndex[NCp])
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time[SPB];
	
	Precision r_State[UD];
	Precision r_Stage1[UD];
	
	
	// K1 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_NextState[0], \
			&r_ActualState[0], \
			s_ActualTime[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_NextState[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + r_NextState[i] * 0.5*s_TimeStep[LocalSystemID];
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + 0.5*s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage1[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage1[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
		{
			r_NextState[i] += 2*r_Stage1[i];
			r_State[i] = r_ActualState[i] + r_Stage1[i]*0.5*s_TimeStep[LocalSystemID];
		}
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage1[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage1[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
		{
			r_NextState[i] += 2*r_Stage1[i];
			r_State[i] = r_ActualState[i] + r_Stage1[i]*s_TimeStep[LocalSystemID];
		}
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage1[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage1[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// NEW STATE --------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
		{
			r_NextState[i] = r_ActualState[i] + s_TimeStep[LocalSystemID]*( r_NextState[i] + r_Stage1[i] )*(1.0/6.0);
			
			if ( isfinite( r_NextState[i] ) == 0 )
				s_IsFinite[LocalSystemID] = 0;
		}
	}
	__syncthreads();
}


// ----------------------------------------------------------------------------
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch_Stepper_RKCK45(\
			const int  LocalSystemID, \
			const int  UnitID, \
			const int  GlobalSystemID, \
			Precision  r_ActualState[UD], \
			Precision  r_NextState[UD], \
			Precision* s_ActualTime, \
			Precision* s_TimeStep, \
			int*       s_IsFinite, \
			Precision  r_Error[UD], \
			Precision  r_UnitParameters[(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:SPB)][(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:SPB)][(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:SPB)][(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[SPB][UPS][NCp], \
			Precision  r_CouplingFactor[NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[SPB][NCp], \
			int        s_CouplingIndex[NCp])
{
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time[SPB];
	
	Precision r_State[UD];
	
	Precision r_Stage1[UD];
	Precision r_Stage2[UD];
	Precision r_Stage3[UD];
	Precision r_Stage4[UD];
	Precision r_Stage5[UD];
	Precision r_Stage6[UD];
	
	
	// K1 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage1[0], \
			&r_ActualState[0], \
			s_ActualTime[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage1[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(1.0/5.0) ) * s_TimeStep[LocalSystemID];
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(1.0/5.0)*s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage2[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage2[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(3.0/40.0) + \
                                              r_Stage2[i] * static_cast<Precision>(9.0/40.0) ) * s_TimeStep[LocalSystemID];
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(3.0/10.0)*s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage3[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage3[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(3.0/10.0) + \
                                              r_Stage2[i] * static_cast<Precision>(-9.0/10.0) + \
											  r_Stage3[i] * static_cast<Precision>(6.0/5.0) ) * s_TimeStep[LocalSystemID];
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(3.0/5.0)*s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage4[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage4[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K5 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(-11.0/54.0) + \
                                              r_Stage2[i] * static_cast<Precision>(5.0/2.0) + \
											  r_Stage3[i] * static_cast<Precision>(-70.0/27.0) + \
											  r_Stage4[i] * static_cast<Precision>(35.0/27.0) ) * s_TimeStep[LocalSystemID];
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage5[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage5[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// K6 ---------------------------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(1631.0/55296.0) + \
                                              r_Stage2[i] * static_cast<Precision>(175.0/512.0) + \
											  r_Stage3[i] * static_cast<Precision>(575.0/13824.0) + \
											  r_Stage4[i] * static_cast<Precision>(44275.0/110592.0) + \
											  r_Stage5[i] * static_cast<Precision>(253.0/4096.0) ) * s_TimeStep[LocalSystemID];
		
		if ( UnitID == 0 )
			s_Time[LocalSystemID] = s_ActualTime[LocalSystemID] + static_cast<Precision>(7.0/8.0)*s_TimeStep[LocalSystemID];
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			UnitID, \
			&r_Stage6[0], \
			&r_State[0], \
			s_Time[LocalSystemID], \
			&r_UnitParameters[0], \
			&s_SystemParameters[LocalSystemID][0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[LocalSystemID][0], \
			&s_IntegerSystemAccessories[LocalSystemID][0], \
			&s_CouplingTerms[LocalSystemID][UnitID][0], \
			&r_CouplingFactor[0]);
	}
	__syncthreads();
	
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<NC; i++)
		{
			Precision CouplingValue = ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches<UPS,SPB,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalSystemID, UnitID, i);
			r_Stage6[ s_CouplingIndex[i] ] += s_CouplingStrength[LocalSystemID][i]*r_CouplingFactor[i]*CouplingValue;
		}
	}
	__syncthreads();
	
	
	// NEW STATE AND ERROR ----------------------------------------------------
	if ( LocalSystemID < SPB )
	{
		for (int i=0; i<UD; i++)
		{
			r_NextState[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(37.0/378.0) + \
                                                  r_Stage3[i] * static_cast<Precision>(250.0/621.0) + \
												  r_Stage4[i] * static_cast<Precision>(125.0/594.0) + \
												  r_Stage6[i] * static_cast<Precision>(512.0/1771.0) ) * s_TimeStep[LocalSystemID];
			
			r_Error[i] = r_Stage1[i] * ( static_cast<Precision>(  37.0/378.0  -  2825.0/27648.0 ) ) + \
                         r_Stage3[i] * ( static_cast<Precision>( 250.0/621.0  - 18575.0/48384.0 ) ) + \
						 r_Stage4[i] * ( static_cast<Precision>( 125.0/594.0  - 13525.0/55296.0 ) ) + \
						 r_Stage5[i] * ( static_cast<Precision>(   0.0        -   277.0/14336.0 ) ) + \
						 r_Stage6[i] * ( static_cast<Precision>( 512.0/1771.0 -     1.0/4.0 ) );
			r_Error[i] = s_TimeStep[LocalSystemID]*abs( r_Error[i] ) + 1e-18;
			
			if ( ( isfinite( r_NextState[i] ) == 0 ) || ( isfinite( r_Error[i] ) == 0 ) )
				s_IsFinite[LocalSystemID] = 0;
		}
	}
	__syncthreads();
}


// --- AUXILIARY FUNCTIONS ---


// COMPUTE COUPLING VALUES
template <int UPS, int SPB, int NC, int NCp, int CBW, int CCI, class Precision>
__forceinline__ __device__ Precision ComputeCouplingValue_MultipleSystems_MultipleBlockLaunches( \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingTerms[SPB][UPS][NCp], \
			int        LocalSystemID, \
			int        UnitID, \
			int        CouplingSerialNumber)
{
	Precision CouplingValue = 0;
	int idx;
	
	// FULL IRREGULAR
	if ( ( CCI == 0 ) && ( CBW == 0 ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS*UPS;
		for (int Col=0; Col<UPS; Col++)
		{
			idx = UnitID + Col*UPS + MemoryShift;
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][Col][CouplingSerialNumber];
		}
	}
	
	// DIAGONAL (including the circular extensions)
	if ( ( CCI == 0 ) && ( CBW > 0  ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS*(2*CBW+1);
		int ShiftedUnitID;
		for (int Diag=0; Diag<(2*CBW+1); Diag++)
		{
			idx = UnitID + Diag*UPS + MemoryShift;
			
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	// CIRCULARLY DIAGONAL
	if ( ( CCI == 1 ) && ( CBW > 0  ) )
	{
		int MemoryShift = CouplingSerialNumber * (2*CBW+1);
		int ShiftedUnitID;
		for (int Diag=0; Diag<(2*CBW+1); Diag++)
		{
			idx = Diag + MemoryShift;
			
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	// FULL CIRCULAR
	if ( ( CCI == 1 ) && ( CBW == 0 ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS;
		int ShiftedUnitID;
		for (int Diag=0; Diag<UPS; Diag++)
		{
			idx = Diag + MemoryShift;
		
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[LocalSystemID][ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	return CouplingValue;
}


template <int UPS, int NC, int NCp, int CBW, int CCI, class Precision>
__forceinline__ __device__ Precision ComputeCouplingValue_SingleSystem_MultipleBlockLaunches( \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingTerms[UPS][NCp], \
			int        UnitID, \
			int        CouplingSerialNumber)
{
	Precision CouplingValue = 0;
	int idx;
	
	// FULL IRREGULAR
	if ( ( CCI == 0 ) && ( CBW == 0 ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS*UPS;
		for (int Col=0; Col<UPS; Col++)
		{
			idx = UnitID + Col*UPS + MemoryShift;
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[Col][CouplingSerialNumber];
		}
	}
	
	// DIAGONAL (including the circular extensions)
	if ( ( CCI == 0 ) && ( CBW > 0  ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS*(2*CBW+1);
		int ShiftedUnitID;
		for (int Diag=0; Diag<(2*CBW+1); Diag++)
		{
			idx = UnitID + Diag*UPS + MemoryShift;
			
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	// CIRCULARLY DIAGONAL
	if ( ( CCI == 1 ) && ( CBW > 0  ) )
	{
		int MemoryShift = CouplingSerialNumber * (2*CBW+1);
		int ShiftedUnitID;
		for (int Diag=0; Diag<(2*CBW+1); Diag++)
		{
			idx = Diag + MemoryShift;
			
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	// FULL CIRCULAR
	if ( ( CCI == 1 ) && ( CBW == 0 ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS;
		int ShiftedUnitID;
		for (int Diag=0; Diag<UPS; Diag++)
		{
			idx = Diag + MemoryShift;
		
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	return CouplingValue;
}

#endif
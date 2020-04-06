#ifndef SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_STEPPERS_H
#define SINGLESYSTEM_PERTHREAD_RUNGEKUTTA_STEPPERS_H


// ----------------------------------------------------------------------------
template <int NBL, int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches_RK4( \
			Precision  r_ActualState[NBL][UD], Precision r_NextState[NBL][UD], Precision* s_ActualTime, Precision* s_TimeStep,\
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
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	for (int BL=0; BL<NBL; BL++)
	{
		if ( ... )
			T  = ActualTime + dTp2;
	}
	
	
	
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + NextState[i] * dTp2;
	
	
	
	
	
	
	/*double X[SD];
	double k1[SD];
	
	double T;
	double dTp2 = 0.5*TimeStep;
	double dTp6 = (1.0/6.0)*TimeStep;
	
	// k1 -----
	PerThread_OdeFunction(tid, NT, NextState, ActualState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// k2 -----
	T  = ActualTime + dTp2;
	#pragma unroll
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + NextState[i] * dTp2;
	
	PerThread_OdeFunction(tid, NT, k1, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// k3 -----
	#pragma unroll
	for (int i=0; i<SD; i++)
	{	
		NextState[i] = NextState[i] + 2*k1[i];
		X[i] = ActualState[i] + k1[i] * dTp2;
	}
	PerThread_OdeFunction(tid, NT, k1, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// k4 -----
	T = ActualTime + TimeStep;
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		NextState[i] = NextState[i] + 2*k1[i];
		X[i] = ActualState[i] + k1[i] * TimeStep;
	}
	PerThread_OdeFunction(tid, NT, k1, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// New state
	#pragma unroll
	for (int i=0; i<SD; i++)
	{
		NextState[i] = ActualState[i] + dTp6 * ( NextState[i] + k1[i] );
		
		if ( isfinite( NextState[i] ) == 0 )
			IsFinite = 0;
	}*/
}


// ----------------------------------------------------------------------------
/*template <int NT, int SD, Algorithms Algorithm>
__forceinline__ __device__ void RungeKuttaStepperRKCK45(int tid, double ActualTime, double TimeStep, double* ActualState, double* NextState, double* Error, bool& IsFinite, double* ControlParameters, double* s_SharedParameters, int* s_IntegerSharedParameters, double* Accessories, int* IntegerAccessories)
{
	double X[SD];
	double T;
	
	double k1[SD];
	double k2[SD];
	double k3[SD];
	double k4[SD];
	double k5[SD];
	double k6[SD];
	
	// k1 -----
	PerThread_OdeFunction(tid, NT, k1, ActualState, ActualTime, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k2 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[0];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[0]*k1[i] );
	
	PerThread_OdeFunction(tid, NT, k2, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k3 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[1];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[2]*k1[i] + d_BT_RKCK45[3]*k2[i] );
	
	PerThread_OdeFunction(tid, NT, k3, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k4 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[4];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[1]*k1[i] + d_BT_RKCK45[5]*k2[i] + d_BT_RKCK45[6]*k3[i] );
	
	PerThread_OdeFunction(tid, NT, k4, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k5 -----
	T = ActualTime + TimeStep;
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[7]*k1[i] + d_BT_RKCK45[8]*k2[i] + d_BT_RKCK45[9]*k3[i] + d_BT_RKCK45[10]*k4[i] );
	
	PerThread_OdeFunction(tid, NT, k5, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	// k6 -----
	T = ActualTime + TimeStep * d_BT_RKCK45[11];
	for (int i=0; i<SD; i++)
		X[i] = ActualState[i] + TimeStep * ( d_BT_RKCK45[12]*k1[i] + d_BT_RKCK45[13]*k2[i] + d_BT_RKCK45[14]*k3[i] + d_BT_RKCK45[15]*k4[i] + d_BT_RKCK45[16]*k5[i] );
	
	PerThread_OdeFunction(tid, NT, k6, X, T, ControlParameters, s_SharedParameters, s_IntegerSharedParameters, Accessories, IntegerAccessories);
	
	
	// New state and error
	for (int i=0; i<SD; i++)
	{
		NextState[i] = ActualState[i] + TimeStep * ( k1[i]*d_BT_RKCK45[17] + k3[i]*d_BT_RKCK45[18] + k4[i]*d_BT_RKCK45[19] + k6[i]*d_BT_RKCK45[20] );
		
		Error[i] = k1[i]*(d_BT_RKCK45[17]-d_BT_RKCK45[21]) + k3[i]*(d_BT_RKCK45[18]-d_BT_RKCK45[22]) + k4[i]*(d_BT_RKCK45[19]-d_BT_RKCK45[23]) - k5[i]*d_BT_RKCK45[24] + k6[i]*(d_BT_RKCK45[20]-d_BT_RKCK45[25]);
		Error[i] = TimeStep * abs( Error[i] ) + 1e-18;
		
		if ( ( isfinite( NextState[i] ) == 0 ) || ( isfinite( Error[i] ) == 0 ) )
			IsFinite = 0;
	}
}*/


#endif
#ifndef PERTHREAD_RUNGEKUTTA_H
#define PERTHREAD_RUNGEKUTTA_H

__constant__ double d_BT_RK4[1];
__constant__ double d_BT_RKCK45[26];

#define SD   KernelParameters.SystemDimension
#define NT   KernelParameters.NumberOfThreads
#define NE   KernelParameters.NumberOfEvents
#define NSP  KernelParameters.NumberOfSharedParameters
#define gTD  KernelParameters.d_TimeDomain
#define gAS  KernelParameters.d_ActualState
#define gPAR KernelParameters.d_ControlParameters
#define gACC KernelParameters.d_Accessories
#define gST  KernelParameters.d_State
#define gSTG KernelParameters.d_Stages
#define gNS  KernelParameters.d_NextState
#define gAEV KernelParameters.d_ActualEventValue
#define gNEV KernelParameters.d_NextEventValue
#define gEC  KernelParameters.d_EventCounter
#define gEQC KernelParameters.d_EventEquilibriumCounter
#define cBT  d_BT_RKCK45


__global__ void PerThread_RKCK45(IntegratorInternalVariables KernelParameters)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int DOI = 0;
	int DOTI;
	int DOSI;
	int i1;
	int i2 = SD*NT;
	
	extern __shared__ int DSM[];
		double* sPAR  = (double*)DSM;
		double* sRTOL = (double*)&sPAR[NSP];
		double* sATOL = (double*)&sRTOL[SD];
		double* sET   = (double*)&sATOL[SD];
		int*    sED   = (int*)&sET[NE];
		int*    sESC  = (int*)&sED[NE];
	
	if (threadIdx.x==0)
	{
		for (int i=0; i<SD; i++)
		{
			sRTOL[i] = __ldg( &KernelParameters.d_RelativeTolerance[i] );
			sATOL[i] = __ldg( &KernelParameters.d_AbsoluteTolerance[i] );
		}
		
		for (int i=0; i<NE; i++)
		{
			sET[i]  = __ldg( &KernelParameters.d_EventTolerance[i] );
			sED[i]  = __ldg( &KernelParameters.d_EventDirection[i] );
			sESC[i] = __ldg( &KernelParameters.d_EventStopCounter[i] );
		}
		
		for (int i=0; i<NSP; i++)
			sPAR[i] = __ldg( &KernelParameters.d_SharedParameters[i] );
	}
	__syncthreads();
	
	double TS = KernelParameters.InitialTimeStep;
	double TE;
	double NTS;
	double TSM;
	double T;
	int MaxNTS = 0;
	
	bool TRM = 0;
	bool UPD;
	bool DUPD = 0;
	bool FIN;
	
	double RER;
	double AER;
	double ETOL;
	
	if (tid < KernelParameters.ActiveThreads)
	{
		double AT  = gTD[tid];
		double TDU = gTD[tid + NT];
		
		// STORE DENSE OUTPUT
		if ( KernelParameters.DenseOutputEnabled == 1 )
		{
			DOTI = tid + DOI*NT;
			KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
			
			DOSI = tid + DOI*NT*SD; i1 = tid;
			for (int i=0; i<SD; i++)
			{
				KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
				DOSI += NT; i1 += NT;
			}
			
			if ( KernelParameters.DenseOutputTimeStep < 0.0 )
				DUPD = 1;
		}
		
		PerThread_EventFunction(tid, NT, gAEV, gAS, AT, gPAR, sPAR, gACC);
		PerThread_Initialization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
		
		i1=tid;
		for (int i=0; i<NE; i++)
		{
			gEC[i1]  = 0;
			gEQC[i1] = 0;
			i1 += NT;
		}
		
		while ( TRM==0 )
		{
			UPD = 1;
			FIN = 1;
			
			if ( TS > (TDU-AT) )
			{	
				TS   = TDU-AT;
				TRM  = 1;
			}
			
			// DENSE OUTPUT TIME STEP CORRECTION
			if ( ( KernelParameters.DenseOutputTimeStep > 0.0 ) && ( KernelParameters.DenseOutputEnabled == 1 ) )
			{
				if ( TS > ( (DOI+1)*KernelParameters.DenseOutputTimeStep+KernelParameters.MinimumTimeStep - AT ) )
				{
					TS   = (DOI+1)*KernelParameters.DenseOutputTimeStep - AT;
					DUPD = 1;
				}
				if ( ( TS > ( (DOI+1)*KernelParameters.DenseOutputTimeStep-KernelParameters.MinimumTimeStep - AT ) ) && \
				     ( TS < ( (DOI+1)*KernelParameters.DenseOutputTimeStep+KernelParameters.MinimumTimeStep - AT ) ) )
				{
					 DUPD = 1;
				}
			}
			
			// RUNGE-KUTTA STEP
			PerThread_OdeFunction(tid, NT, &gSTG[0], gAS, AT, gPAR, sPAR, gACC);
			
			T  = AT + TS * cBT[0];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[0]*gSTG[i1] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[i2], gST, T, gPAR, sPAR, gACC);
			
			T  = AT + TS * cBT[1];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[2]*gSTG[i1] + cBT[3]*gSTG[i1+i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[2*i2], gST, T, gPAR, sPAR, gACC);
			
			T  = AT + TS * cBT[4];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[1]*gSTG[i1] + cBT[5]*gSTG[i1+i2] + cBT[6]*gSTG[i1+2*i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[3*i2], gST, T, gPAR, sPAR, gACC);
			
			T  = AT + TS;
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[7]*gSTG[i1] + cBT[8]*gSTG[i1+i2] + cBT[9]*gSTG[i1+2*i2] + cBT[10]*gSTG[i1+3*i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[4*i2], gST, T, gPAR, sPAR, gACC);

			T  = AT + TS * cBT[11];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[12]*gSTG[i1] + cBT[13]*gSTG[i1+i2] + cBT[14]*gSTG[i1+2*i2] + cBT[15]*gSTG[i1+3*i2] + cBT[16]*gSTG[i1+4*i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[5*i2], gST, T, gPAR, sPAR, gACC);
			
			// ERROR HANDLING
			i1 = tid;
			RER = 1e30;
			for (int i=0; i<SD; i++)
			{
				gNS[i1] = gAS[i1] + TS * ( gSTG[i1]*cBT[17] + gSTG[i1+2*i2]*cBT[18] + gSTG[i1+3*i2]*cBT[19] + gSTG[i1+5*i2]*cBT[20] );
				
				AER  = gSTG[i1]*(cBT[17]-cBT[21]) + gSTG[i1+2*i2]*(cBT[18]-cBT[22]) + gSTG[i1+3*i2]*(cBT[19]-cBT[23]) - gSTG[i1+4*i2]*cBT[24] + gSTG[i1+5*i2]*(cBT[20]-cBT[25]);
				AER  = TS * abs( AER );
				ETOL = fmax( sRTOL[i] * abs(gAS[i1]), sATOL[i] );
				
				UPD = UPD && ( AER<ETOL );
				RER = fmin( RER, ETOL / AER );
				
				if ( isfinite(gNS[i1]) == 0 )
					FIN = 0;
				
				i1 += NT;
			}
			
			if (UPD == 1)
				TSM = 0.9 * pow(RER, cBT[0] );
			else
				TSM = 0.9 * pow(RER, cBT[25] );
			
			if ( isfinite(TSM) == 0 )
				FIN = 0;
			
			if ( FIN == 0 )
			{
				if ( TS<(KernelParameters.MinimumTimeStep*1.001) )
				{
					printf("Error: State is not a finite number even with the minimal step size. Try to use less stringent tolerances. (thread id: %d)\n", tid);
					TRM = 1;
				}
				TSM = KernelParameters.TimeStepShrinkLimit;
				UPD = 0;
			} else
			{
				if ( TS<(KernelParameters.MinimumTimeStep*1.001) )
				{
					printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, TS, KernelParameters.MinimumTimeStep);
					UPD = 1;
				}
			}
			
			TSM = fmin(TSM, KernelParameters.TimeStepGrowLimit);
			TSM = fmax(TSM, KernelParameters.TimeStepShrinkLimit);
			
			NTS = TS * TSM;
			
			NTS = fmin(NTS, KernelParameters.MaximumTimeStep);
			NTS = fmax(NTS, KernelParameters.MinimumTimeStep);
			
			// EVENT HANDLING
			PerThread_EventFunction(tid, NT, gNEV, gNS, AT, gPAR, sPAR, gACC);
			
			if ( UPD == 1 )
			{
				TE = TS;
				i1 = tid;
				for (int i=0; i<NE; i++)
				{
					if ( ( ( gAEV[i1] >  sET[i] ) && ( gNEV[i1] < -sET[i] ) && ( sED[i] <= 0 ) ) || \
						( ( gAEV[i1] < -sET[i] ) && ( gNEV[i1] >  sET[i] ) && ( sED[i] >= 0 ) ) )
					{
						TE = fmin( TE, -gAEV[i1] / (gNEV[i1]-gAEV[i1]) * TS );
						UPD = 0;
						
						if ( TE<(KernelParameters.MinimumTimeStep*1.001) )
						{
							printf("Warning: Event cannot be detected without reducing the step size below the minimum! Event detection omitted!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, TE, KernelParameters.MinimumTimeStep);
							UPD = 1;
							break;
						}
					}
					i1 += NT;
				}
				
				if ( UPD == 0 )
					NTS = TE;
			}
			
			// UPDATE STATE
			if ( UPD == 1 )
			{
				AT = AT + TS;
				
				i1 = tid;
				for (int i=0; i<SD; i++)
				{
					gAS[i1]  = gNS[i1];
					i1 += NT;
				}
				
				// STORE DENSE OUTPUT
				if ( ( DUPD == 1 ) && ( KernelParameters.DenseOutputEnabled == 1 ) && ( DOI<KernelParameters.DenseOutputNumberOfPoints ) )
				{
					DOI++;
					
					KernelParameters.d_DenseOutputIndex[tid] = DOI;
					
					DOTI = tid + DOI*NT;
					KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
					
					DOSI = tid + DOI*NT*SD; i1 = tid;
					for (int i=0; i<SD; i++)
					{
						KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
						DOSI += NT; i1 += NT;
					}
					
					if ( KernelParameters.DenseOutputTimeStep > 0.0 )
						DUPD = 0;
				}
				
				i1 = tid;
				for (int i=0; i<NE; i++)
				{
					if ( ( ( gAEV[i1] >  sET[i] ) && ( abs(gNEV[i1]) < sET[i] ) && ( sED[i] <= 0 ) ) || \
					     ( ( gAEV[i1] < -sET[i] ) && ( abs(gNEV[i1]) < sET[i] ) && ( sED[i] >= 0 ) ) )
					{
						gEC[i1]++;
						if ( gEC[i1] == sESC[i] )
							TRM = 1;
						
						PerThread_ActionAfterEventDetection(tid, NT, i, gEC[i1], AT, TS, gTD, gAS, gPAR, sPAR, gACC);
						PerThread_EventFunction(tid, NT, gNEV, gAS, AT, gPAR, sPAR, gACC);
					}
					
					if ( ( abs(gAEV[i1]) <  sET[i] ) && ( abs(gNEV[i1]) > sET[i] ) )
						gEQC[i1] = 0;
					
					if ( ( abs(gAEV[i1]) <  sET[i] ) && ( abs(gNEV[i1]) < sET[i] ) )
						gEQC[i1]++;
					
					if ( gEQC[i1] == KernelParameters.MaxStepInsideEvent)
						TRM = 1;
					
					gAEV[i1] = gNEV[i1];
					i1 += NT;
				}
				
				MaxNTS++;
				if ( ( KernelParameters.MaximumNumberOfTimeSteps != 0 ) && ( MaxNTS >= KernelParameters.MaximumNumberOfTimeSteps ) )
					TRM  = 1;
				
				PerThread_ActionAfterSuccessfulTimeStep(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
			}
			
			TS = NTS;
		}
		
		PerThread_Finalization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
	}
}


/*__global__ void PerThread_RKCK45_EH0(IntegratorInternalVariables KernelParameters)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int DOI = 0;
	int DOTI;
	int DOSI;
	int i1;
	int i2 = SD*NT;
	
	extern __shared__ int DSM[];
		double* sPAR  = (double*)DSM;
		double* sRTOL = (double*)&sPAR[NSP];
		double* sATOL = (double*)&sRTOL[SD];
	
	if (threadIdx.x==0)
	{
		for (int i=0; i<SD; i++)
		{
			sRTOL[i] = __ldg( &KernelParameters.d_RelativeTolerance[i] );
			sATOL[i] = __ldg( &KernelParameters.d_AbsoluteTolerance[i] );
		}
		
		for (int i=0; i<NSP; i++)
			sPAR[i] = __ldg( &KernelParameters.d_SharedParameters[i] );
	}
	__syncthreads();
	
	double TS = KernelParameters.InitialTimeStep;
	double TSM;
	double T;
	int MaxNTS = 0;
	
	bool TRM = 0;
	bool UPD;
	bool DUPD = 0;
	bool FIN;
	
	double RER;
	double AER;
	double ETOL;
	
	if (tid < KernelParameters.ActiveThreads)
	{
		double AT  = gTD[tid];
		double TDU = gTD[tid + NT];
		
		// STORE DENSE OUTPUT
		if ( KernelParameters.DenseOutputEnabled == 1 )
		{
			DOTI = tid + DOI*NT;
			KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
			
			DOSI = tid + DOI*NT*SD; i1 = tid;
			for (int i=0; i<SD; i++)
			{
				KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
				DOSI += NT; i1 += NT;
			}
			
			if ( KernelParameters.DenseOutputTimeStep < 0.0 )
				DUPD = 1;
		}
		
		PerThread_Initialization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
		
		while ( TRM==0 )
		{
			UPD = 1;
			FIN = 1;
			
			if ( TS > (TDU-AT) )
			{	
				TS   = TDU-AT;
				TRM  = 1;
			}
			
			// DENSE OUTPUT TIME STEP CORRECTION
			if ( ( KernelParameters.DenseOutputTimeStep > 0.0 ) && ( KernelParameters.DenseOutputEnabled == 1 ) )
			{
				if ( TS > ( (DOI+1)*KernelParameters.DenseOutputTimeStep+KernelParameters.MinimumTimeStep - AT ) )
				{
					TS   = (DOI+1)*KernelParameters.DenseOutputTimeStep - AT;
					DUPD = 1;
				}
				if ( ( TS > ( (DOI+1)*KernelParameters.DenseOutputTimeStep-KernelParameters.MinimumTimeStep - AT ) ) && \
				     ( TS < ( (DOI+1)*KernelParameters.DenseOutputTimeStep+KernelParameters.MinimumTimeStep - AT ) ) )
				{
					 DUPD = 1;
				}
			}
			
			// RUNGE-KUTTA STEP
			PerThread_OdeFunction(tid, NT, &gSTG[0], gAS, AT, gPAR, sPAR, gACC);
			
			T  = AT + TS * cBT[0];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[0]*gSTG[i1] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[i2], gST, T, gPAR, sPAR, gACC);
			
			T  = AT + TS * cBT[1];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[2]*gSTG[i1] + cBT[3]*gSTG[i1+i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[2*i2], gST, T, gPAR, sPAR, gACC);
			
			T  = AT + TS * cBT[4];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[1]*gSTG[i1] + cBT[5]*gSTG[i1+i2] + cBT[6]*gSTG[i1+2*i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[3*i2], gST, T, gPAR, sPAR, gACC);
			
			T  = AT + TS;
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[7]*gSTG[i1] + cBT[8]*gSTG[i1+i2] + cBT[9]*gSTG[i1+2*i2] + cBT[10]*gSTG[i1+3*i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[4*i2], gST, T, gPAR, sPAR, gACC);

			T  = AT + TS * cBT[11];
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + TS * ( cBT[12]*gSTG[i1] + cBT[13]*gSTG[i1+i2] + cBT[14]*gSTG[i1+2*i2] + cBT[15]*gSTG[i1+3*i2] + cBT[16]*gSTG[i1+4*i2] );
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, &gSTG[5*i2], gST, T, gPAR, sPAR, gACC);
			
			// ERROR HANDLING
			i1 = tid;
			RER = 1e30;
			for (int i=0; i<SD; i++)
			{
				gNS[i1] = gAS[i1] + TS * ( gSTG[i1]*cBT[17] + gSTG[i1+2*i2]*cBT[18] + gSTG[i1+3*i2]*cBT[19] + gSTG[i1+5*i2]*cBT[20] );
				
				AER  = gSTG[i1]*(cBT[17]-cBT[21]) + gSTG[i1+2*i2]*(cBT[18]-cBT[22]) + gSTG[i1+3*i2]*(cBT[19]-cBT[23]) - gSTG[i1+4*i2]*cBT[24] + gSTG[i1+5*i2]*(cBT[20]-cBT[25]);
				AER  = TS * abs( AER );
				ETOL = fmax( sRTOL[i] * abs(gAS[i1]), sATOL[i] );
				
				UPD = UPD && ( AER<ETOL );
				RER = fmin( RER, ETOL / AER );
				
				if ( isfinite(gNS[i1]) == 0 )
					FIN = 0;
				
				i1 += NT;
			}
			
			if (UPD == 1)
				TSM = 0.9 * pow(RER, cBT[0] );
			else
				TSM = 0.9 * pow(RER, cBT[25] );
			
			if ( isfinite(TSM) == 0 )
				FIN = 0;
			
			if ( FIN == 0 )
			{
				if ( TS<(KernelParameters.MinimumTimeStep*1.001) )
				{
					printf("Error: State is not a finite number even with the minimal step size. Try to use less stringent tolerances. (thread id: %d)\n", tid);
					TRM = 1;
				}
				TSM = KernelParameters.TimeStepShrinkLimit;
				UPD = 0;
			} else
			{
				if ( TS<(KernelParameters.MinimumTimeStep*1.001) )
				{
					printf("Warning: Minimum step size reached! Continue with fixed minimum step size! Tolerance cannot be guaranteed!, thread id: %d, time step: %+6.5e, min step size: %+6.5e \n", tid, TS, KernelParameters.MinimumTimeStep);
					UPD = 1;
				}
			}
			
			TSM = fmin(TSM, KernelParameters.TimeStepGrowLimit);
			TSM = fmax(TSM, KernelParameters.TimeStepShrinkLimit);
			
			// UPDATE STATE
			if ( UPD == 1 )
			{
				AT = AT + TS;
				
				i1 = tid;
				for (int i=0; i<SD; i++)
				{
					gAS[i1] = gNS[i1];
					i1 += NT;
				}
				
				// STORE DENSE OUTPUT
				if ( ( DUPD == 1 ) && ( KernelParameters.DenseOutputEnabled == 1 ) && ( DOI<KernelParameters.DenseOutputNumberOfPoints ) )
				{
					DOI++;
					
					KernelParameters.d_DenseOutputIndex[tid] = DOI;
					
					DOTI = tid + DOI*NT;
					KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
					
					DOSI = tid + DOI*NT*SD; i1 = tid;
					for (int i=0; i<SD; i++)
					{
						KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
						DOSI += NT; i1 += NT;
					}
					
					if ( KernelParameters.DenseOutputTimeStep > 0.0 )
						DUPD = 0;
				}
				
				MaxNTS++;
				if ( ( KernelParameters.MaximumNumberOfTimeSteps != 0 ) && ( MaxNTS >= KernelParameters.MaximumNumberOfTimeSteps ) )
					TRM  = 1;
				
				PerThread_ActionAfterSuccessfulTimeStep(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
			}
			
			TS = TS * TSM;
			TS = fmin(TS, KernelParameters.MaximumTimeStep);
			TS = fmax(TS, KernelParameters.MinimumTimeStep);
		}
		
		PerThread_Finalization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
	}
}


__global__ void PerThread_RK4(IntegratorInternalVariables KernelParameters)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int DOI = 0;
	int DOTI;
	int DOSI;
	int i1;
	
	extern __shared__ int DSM[];
		double* sPAR = (double*)DSM;
		double* sET  = (double*)&sPAR[NSP];
		int*    sED  = (int*)&sET[NE];
		int*    sESC = (int*)&sED[NE];
	
	if (threadIdx.x==0)
	{
		for (int i=0; i<NE; i++)
		{
			sET[i]  = __ldg( &KernelParameters.d_EventTolerance[i] );
			sED[i]  = __ldg( &KernelParameters.d_EventDirection[i] );
			sESC[i] = __ldg( &KernelParameters.d_EventStopCounter[i] );
		}
		
		for (int i=0; i<NSP; i++)
			sPAR[i] = __ldg( &KernelParameters.d_SharedParameters[i] );
	}
	__syncthreads();
	
	double TS   = KernelParameters.InitialTimeStep;
	double TSp2 = KernelParameters.InitialTimeStep * 0.5;
	double TE;
	double T;
	int MaxNTS = 0;
	
	bool TRM = 0;
	bool UPD;
	
	if (tid < KernelParameters.ActiveThreads)
	{
		double AT  = gTD[tid];
		double TDU = gTD[tid + NT];
		
		// STORE DENSE OUTPUT
		if ( KernelParameters.DenseOutputEnabled == 1 )
		{
			DOTI = tid + DOI*NT;
			KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
			
			DOSI = tid + DOI*NT*SD; i1 = tid;
			for (int i=0; i<SD; i++)
			{
				KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
				DOSI += NT; i1 += NT;
			}
		}
		
		PerThread_EventFunction(tid, NT, gAEV, gAS, AT, gPAR, sPAR, gACC);
		PerThread_Initialization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
		
		i1=tid;
		for (int i=0; i<NE; i++)
		{
			gEC[i1]  = 0;
			gEQC[i1] = 0;
			i1 += NT;
		}
		
		while ( TRM==0 )
		{
			UPD = 1;
			
			if ( TS > (TDU-AT) )
			{	
				TS   = TDU-AT;
				TSp2 = TS * 0.5;
				TRM  = 1;
			}
			
			// RUNGE-KUTTA STEP
			PerThread_OdeFunction(tid, NT, gNS, gAS, AT, gPAR, sPAR, gACC);
			
			T  = AT + TSp2;
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + gNS[i1] * TSp2;
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, gSTG, gST, T, gPAR, sPAR, gACC);
			
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gNS[i1] = gNS[i1] + 2*gSTG[i1];
				gST[i1] = gAS[i1] +   gSTG[i1] * TSp2;
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, gSTG, gST, T, gPAR, sPAR, gACC);
			
			T = AT + TS;
			i1 = tid;
			for (int i=0; i<SD; i++)
			{
				gNS[i1] = gNS[i1] + 2*gSTG[i1];
				gST[i1] = gAS[i1] +   gSTG[i1] * TS;
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, gSTG, gST, T, gPAR, sPAR, gACC);
			
			i1 = tid;
			for (int i=0; i<SD; i++)
			{
				gNS[i1] = gAS[i1] + TS*d_BT_RK4[0] * ( gNS[i1] + gSTG[i1] );
				if ( isfinite( gNS[i1] ) == 0 )
				{
					printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
					TRM = 1;
				}
				i1 += NT;
			}
			
			// EVENT HANDLING
			PerThread_EventFunction(tid, NT, gNEV, gNS, AT, gPAR, sPAR, gACC);
			
			TE = TS;
			i1 = tid;
			for (int i=0; i<NE; i++)
			{
				if ( ( ( gAEV[i1] >  sET[i] ) && ( gNEV[i1] < -sET[i] ) && ( sED[i] <= 0 ) ) || \
				     ( ( gAEV[i1] < -sET[i] ) && ( gNEV[i1] >  sET[i] ) && ( sED[i] >= 0 ) ) )
				{
					TE = fmin( TE, -gAEV[i1] / (gNEV[i1]-gAEV[i1]) * TS );
					UPD = 0;
				}
				i1 += NT;
			}
			TS   = TE;
			TSp2 = TS * 0.5;
			
			// UPDATE STATE
			if ( UPD == 1 )
			{
				AT = AT + TS;
				
				i1 = tid;
				for (int i=0; i<SD; i++)
				{
					gAS[i1]  = gNS[i1];
					i1 += NT;
				}
				
				// STORE DENSE OUTPUT
				DOI++;
				if ( ( KernelParameters.DenseOutputEnabled == 1 ) && ( DOI<KernelParameters.DenseOutputNumberOfPoints ) )
				{
					KernelParameters.d_DenseOutputIndex[tid] = DOI;
					
					DOTI = tid + DOI*NT;
					KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
					
					DOSI = tid + DOI*NT*SD; i1 = tid;
					for (int i=0; i<SD; i++)
					{
						KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
						DOSI += NT; i1 += NT;
					}
				}
				
				// EVENT HANDLING
				i1 = tid;
				for (int i=0; i<NE; i++)
				{
					if ( ( ( gAEV[i1] >  sET[i] ) && ( abs(gNEV[i1]) < sET[i] ) && ( sED[i] <= 0 ) ) || \
					     ( ( gAEV[i1] < -sET[i] ) && ( abs(gNEV[i1]) < sET[i] ) && ( sED[i] >= 0 ) ) )
					{
						gEC[i1]++;
						if ( gEC[i1] == sESC[i] )
							TRM = 1;
						
						TS   = KernelParameters.InitialTimeStep;
						
						PerThread_ActionAfterEventDetection(tid, NT, i, gEC[i1], AT, TS, gTD, gAS, gPAR, sPAR, gACC);
						PerThread_EventFunction(tid, NT, gNEV, gAS, AT, gPAR, sPAR, gACC);
						
						TSp2 = TS * 0.5;
					}
					
					if ( ( abs(gAEV[i1]) <  sET[i] ) && ( abs(gNEV[i1]) > sET[i] ) )
						gEQC[i1] = 0;
					
					if ( ( abs(gAEV[i1]) <  sET[i] ) && ( abs(gNEV[i1]) < sET[i] ) )
						gEQC[i1]++;
					
					if ( gEQC[i1] == KernelParameters.MaxStepInsideEvent )
						TRM = 1;
					
					gAEV[i1] = gNEV[i1];
					i1 += NT;
				}
				
				MaxNTS++;
				if ( ( KernelParameters.MaximumNumberOfTimeSteps != 0 ) && ( MaxNTS >= KernelParameters.MaximumNumberOfTimeSteps ) )
					TRM  = 1;
				
				PerThread_ActionAfterSuccessfulTimeStep(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
			}
		}
		
		PerThread_Finalization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
	}
}


__global__ void PerThread_RK4_EH0(IntegratorInternalVariables KernelParameters)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int DOI = 0;
	int DOTI;
	int DOSI;
	int i1;
	
	extern __shared__ int DSM[];
		double* sPAR = (double*)DSM;
	
	if (threadIdx.x==0)
	{
		for (int i=0; i<NSP; i++)
			sPAR[i] = __ldg( &KernelParameters.d_SharedParameters[i] );
	}
	__syncthreads();
	
	double TS   = KernelParameters.InitialTimeStep;
	double TSp2 = KernelParameters.InitialTimeStep * 0.5;
	double T;
	int MaxNTS = 0;
	
	bool TRM = 0;
	
	if (tid < KernelParameters.ActiveThreads)
	{
		double AT  = gTD[tid];
		double TDU = gTD[tid + NT];
		
		// STORE DENSE OUTPUT
		if ( KernelParameters.DenseOutputEnabled == 1 )
		{
			DOTI = tid + DOI*NT;
			KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
			
			DOSI = tid + DOI*NT*SD; i1 = tid;
			for (int i=0; i<SD; i++)
			{
				KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
				DOSI += NT; i1 += NT;
			}
		}
		
		PerThread_Initialization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
		
		while ( TRM==0 )
		{
			if ( TS > (TDU-AT) )
			{	
				TS   = TDU-AT;
				TSp2 = TS * 0.5;
				TRM  = 1;
			}
			
			// RUNGE-KUTTA STEP
			PerThread_OdeFunction(tid, NT, gNS, gAS, AT, gPAR, sPAR, gACC);
			
			T  = AT + TSp2;
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gST[i1] = gAS[i1] + gNS[i1] * TSp2;
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, gSTG, gST, T, gPAR, sPAR, gACC);
			
			i1 = tid;
			for (int i=0; i<SD; i++)
			{	
				gNS[i1] = gNS[i1] + 2*gSTG[i1];
				gST[i1] = gAS[i1] +   gSTG[i1] * TSp2;
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, gSTG, gST, T, gPAR, sPAR, gACC);
			
			T = AT + TS;
			i1 = tid;
			for (int i=0; i<SD; i++)
			{
				gNS[i1] = gNS[i1] + 2*gSTG[i1];
				gST[i1] = gAS[i1] +   gSTG[i1] * TS;
				i1 += NT;
			}
			PerThread_OdeFunction(tid, NT, gSTG, gST, T, gPAR, sPAR, gACC);
			
			
			// UPDATE STATE
			i1 = tid;
			for (int i=0; i<SD; i++)
			{
				gAS[i1] = gAS[i1] + TS*d_BT_RK4[0] * ( gNS[i1] + gSTG[i1] );
				if ( isfinite( gAS[i1] ) == 0 )
				{
					printf("Error: State is not a finite number. Try to use smaller step size. (thread id: %d)\n", tid);
					TRM = 1;
				}
				i1 += NT;
			}
			AT = AT + TS;
			
			// STORE DENSE OUTPUT
			DOI++;
			if ( ( KernelParameters.DenseOutputEnabled == 1 ) && ( DOI<KernelParameters.DenseOutputNumberOfPoints ) )
			{
				KernelParameters.d_DenseOutputIndex[tid] = DOI;
				
				DOTI = tid + DOI*NT;
				KernelParameters.d_DenseOutputTimeInstances[DOTI] = AT;
				
				DOSI = tid + DOI*NT*SD; i1 = tid;
				for (int i=0; i<SD; i++)
				{
					KernelParameters.d_DenseOutputStates[DOSI] = gAS[i1];
					DOSI += NT; i1 += NT;
				}
			}
			
			MaxNTS++;
			if ( ( KernelParameters.MaximumNumberOfTimeSteps != 0 ) && ( MaxNTS >= KernelParameters.MaximumNumberOfTimeSteps ) )
				TRM  = 1;
			
			PerThread_ActionAfterSuccessfulTimeStep(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
		}
		
		PerThread_Finalization(tid, NT, AT, TS, gTD, gAS, gPAR, sPAR, gACC);
	}
}*/

#undef SD
#undef NT
#undef NE
#undef NSP
#undef gTD
#undef gAS
#undef gPAR
#undef sPAR
#undef gACC
#undef gST
#undef gSTG
#undef gNS
#undef gAEV
#undef gNEV
#undef gEC
#undef gEQC
#undef cBT

#endif
	// Testing thread management
	for (int BL=0; BL<NumberOfBlockLaunches; BL++)
	{
		bool LimitReached;
		LocalThreadID_Logical  = LocalThreadID_GPU + BL*blockDim.x;
		GlobalThreadID_Logical = LocalThreadID_Logical + BlockID*TotalLogicalThreadsPerBlock;
		
		LocalSystemID  = LocalThreadID_Logical / UPS;
		UnitID         = LocalThreadID_Logical % UPS;
		GlobalSystemID = LocalSystemID + BlockID*SPB;
		
		if ( ( LocalSystemID >= SPB ) || ( GlobalSystemID >= NS ) || ( GlobalSystemID >= SolverOptions.ActiveSystems ) )
			LimitReached = 1;
		else
			LimitReached = 0;
		
		printf("GlbTID_Log: %d, GlbTID_GPU: %d, BlockID: %d, LocTID_Log: %d, LocTID_GPU: %d, GlbSID: %d, LocSID: %d, UnitID: %d, LIMIT: %d \n", \
		GlobalThreadID_Logical, GlobalThreadID_GPU, BlockID, LocalThreadID_Logical, LocalThreadID_GPU, GlobalSystemID, LocalSystemID, UnitID, LimitReached);
	}
	
	// Testing global shared memory variables
	if ( ( blockIdx.x == 0 ) && ( threadIdx.x == 0 ) )
	{
		for (int i=0; i<NC; i++)
			printf("Serial number: %d, s_CouplingIndex: %d \n", i, s_CouplingIndex[i]);
		
		for (int i=0; i<NGP; i++)
			printf("Serial number: %d, gs_GlobalParameters: %6.3e \n", i, gs_GlobalParameters[i]);
	}
	
	// Testing shared memory management
	if ( threadIdx.x == 0 )
	{
		/*for (int i=0; i<SPB; i++) // No limit check is necessary due to SYSTEM PADDING
			printf("System number: %d, TimeDomain[0]: %6.3e, TimeDomain[1]: %6.3e \n", i, s_TimeDomain[i][0], s_TimeDomain[i][1]);
		
		for (int i=0; i<SPB; i++)
			printf("System number: %d, SystemParameters[0]: %6.3e \n", i, s_SystemParameters[i][0]);
		
		for (int i=0; i<SPB; i++)
			printf("System number: %d, SystemAccessories[0]: %6.3e, SystemAccessories[1]: %6.3e, SystemAccessories[2]: %6.3e \n", i, s_SystemAccessories[i][0], s_SystemAccessories[i][1], s_SystemAccessories[i][2]);
		
		for (int i=0; i<SPB; i++)
			printf("System number: %d, IntegerSystemAccessories[0]: %d, IntegerSystemAccessories[1]: %d, IntegerSystemAccessories[2]: %d, IntegerSystemAccessories[3]: %d \n", i, s_IntegerSystemAccessories[i][0], s_IntegerSystemAccessories[i][1], s_IntegerSystemAccessories[i][2], s_IntegerSystemAccessories[i][3]);
	
		for (int i=0; i<NGP; i++)
			printf("Serial number: %d, gs_GlobalParameters: %6.3e \n", i, gs_GlobalParameters[i]);*/
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, gs_IntegerGlobalParameters[%d]: %d \n", BlockID, i + BlockID*SPB, i, 5, gs_IntegerGlobalParameters[5]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_RelativeTolerance[0]: %6.3e, s_RelativeTolerance[1]: %6.3e \n", BlockID, i + BlockID*SPB, i, s_RelativeTolerance[0], s_RelativeTolerance[1]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_AbsoluteTolerance[0]: %6.3e, s_AbsoluteTolerance[1]: %6.3e \n", BlockID, i + BlockID*SPB, i, s_AbsoluteTolerance[0], s_AbsoluteTolerance[1]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_EventTolerance[0]: %6.3e, s_EventTolerance[1]: %6.3e, s_EventTolerance[2]: %6.3e \n", BlockID, i + BlockID*SPB, i, s_EventTolerance[0], s_EventTolerance[1], s_EventTolerance[2]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_EventStopCounter[0]: %d, s_EventStopCounter[1]: %d, s_EventStopCounter[2]: %d \n", BlockID, i + BlockID*SPB, i, s_EventDirection[0], s_EventDirection[1], s_EventDirection[2]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_EventStopCounter[0]: %d, s_EventStopCounter[1]: %d, s_EventStopCounter[2]: %d \n", BlockID, i + BlockID*SPB, i, s_EventStopCounter[0], s_EventStopCounter[1], s_EventStopCounter[2]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_CouplingStrength[0]: %6.3e, s_CouplingStrength[1]: %6.3e \n", BlockID, i + BlockID*SPB, i, s_CouplingStrength[i][0], s_CouplingStrength[i][1]);
		
		//for (int i=0; i<SPB; i++)
		//	printf("BlockID: %d, Global system number: %d, Local system number: %d, s_DenseOutputIndex: %d \n", BlockID, i + BlockID*SPB, i, s_DenseOutputIndex[i]);
	}
	
	// Testing coupling matrices
	if ( ( BlockID == 1 ) && ( threadIdx.x == 0 ) )
	{
		int SerialNumber = 0;
		int MemoryShift  = SerialNumber*UPS*UPS;
		int idx;
		for (int row=0; row<UPS; row++)
		{
			for (int col=0; col<UPS; col++)
			{
				idx = row + col*UPS + MemoryShift;
				printf("%6.3e ", gs_CouplingMatrix[idx]);
			}
			printf("\n");
		}
	}
	
	
	// Testing register memory management
	if ( ( BlockID == 2 ) && ( threadIdx.x == 0 ) )
	{
		/*for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			printf("BL: %d, X1: %6.3e, X2: %6.3e \n", BL, r_ActualState[BL][0], r_ActualState[BL][1]);
		
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			printf("BL: %d, UP1: %6.3e, UP2: %6.3e, UP3: %6.3e, UP4: %6.3e \n", BL, r_UnitParameters[BL][0], r_UnitParameters[BL][1], r_UnitParameters[BL][2], r_UnitParameters[BL][3]);
		
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			printf("BL: %d, UA1: %6.3e \n", BL, r_UnitAccessories[BL][0]);
		
		for (int BL=0; BL<NumberOfBlockLaunches; BL++)
			printf("BL: %d, iUA1: %d, iUA2: %d \n", BL, r_IntegerUnitAccessories[BL][0], r_IntegerUnitAccessories[BL][1]);*/
	}
	
	// Testing shared memory splitting
	/*Precision* pr_ActualState = &r_ActualState[0][0];
	if ( ( BlockID == 0 ) && ( threadIdx.x == 4 ) )
	{
		for (int i=0; i<NumberOfBlockLaunches*UD; i++)
		{
			printf("p_X: %6.3e \n", pr_ActualState[i]);
		}
		printf("\n");
	}
	
	pr_ActualState = &r_ActualState[0][0];
	if ( ( BlockID == 0 ) && ( threadIdx.x == 4 ) )
	{
		for (int i=0; i<UD; i++)
		{
			printf("p_X: %6.3e \n", pr_ActualState[i]);
		}
		printf("\n");
	}
	
	pr_ActualState = &r_ActualState[1][0];
	if ( ( BlockID == 0 ) && ( threadIdx.x == 4 ) )
	{
		for (int i=0; i<UD; i++)
		{
			printf("p_X: %6.3e \n", pr_ActualState[i]);
		}
		printf("\n");
	}*/
	
	
	// Complex test of ODE function output-------------------------------------
	int GlobalSystemID;
	int GSID = 1;
	int UID  = 2;
	for (int BL=0; BL<NBL; BL++)
	{
		int CouplingIndex = 0;
		
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		GlobalSystemID        = LocalSystemID + blockIdx.x*SPB;
		
		// All
		/*if ( (LocalSystemID<SPB) && (GlobalSystemID<NS) )
		{
			printf("BL: %2d, GSID: %2d, LSID: %2d, UID: %2d, X1: %+6.3e, X2: %+6.3e, F1: %+6.3e, F2: %+6.3e, CPT: %+6.3e, CPF: %+6.3e \n", \
			    BL, GlobalSystemID, LocalSystemID, UnitID, \
				r_ActualState[BL][0], r_ActualState[BL][1], r_NextState[BL][0], r_NextState[BL][1], \
				s_CouplingTerms[LocalSystemID][UnitID][CouplingIndex], r_CouplingFactor[BL][CouplingIndex]);
		}*/
		
		// Specific
		if ( (GlobalSystemID==GSID) && (UnitID==UID) && (LocalSystemID<SPB) )
		{
			printf("BL: %2d, GSID: %2d, LSID: %2d, UID: %2d, X1: %+6.3e, X2: %+6.3e, F1: %+6.3e, F2: %+6.3e, CPT: %+6.3e, CPF: %+6.3e \n", \
			    BL, GlobalSystemID, LocalSystemID, UnitID, \
				r_ActualState[BL][0], r_ActualState[BL][1], r_NextState[BL][0], r_NextState[BL][1], \
				s_CouplingTerms[LocalSystemID][UnitID][CouplingIndex], r_CouplingFactor[BL][CouplingIndex]);
		}
		
		// Corresponding coupling terms
		if ( (GlobalSystemID==GSID) && (LocalSystemID<SPB) )
		{
			printf("BL: %2d, GSID: %2d, LSID: %2d, UID: %2d, s_CouplingTerms: %+6.3e \n", \
			    BL, GlobalSystemID, LocalSystemID, UnitID, \
				s_CouplingTerms[LocalSystemID][UnitID][CouplingIndex]);
		}
	}
	__syncthreads(); //--------------------------------------------------------
	
				// Test coupling value accumulation (embedded in the Matrix-CouplinTerms multiplication)
				GlobalSystemID = LocalSystemID + blockIdx.x*SPB;
				if ( (GlobalSystemID==GSID) && (LocalSystemID<SPB) && (UnitID==UID) && (i==0) )
				{
					printf("GSID: %2d, LSID: %2d, NC: %2d, Row: %2d, Col: %2d, CouplingMatrix: %+6.3e, CouplingTerms: %+6.3e, CouplingValue: %+6.3e, \n", \
						LocalSystemID + blockIdx.x*SPB, LocalSystemID, i, Row, Col, gs_CouplingMatrix[idx], s_CouplingTerms[LocalSystemID][Col][i], CouplingValue);
				}
	
	// Test ODE function output after coupling --------------------------------
	for (int BL=0; BL<NBL; BL++)
	{
		int CouplingIndex = 0;
		
		LocalThreadID_Logical = LocalThreadID_GPU + BL*blockDim.x;
		LocalSystemID         = LocalThreadID_Logical / UPS;
		UnitID                = LocalThreadID_Logical % UPS;
		GlobalSystemID        = LocalSystemID + blockIdx.x*SPB;
		
		// All
		/*if ( (LocalSystemID<SPB) && (GlobalSystemID<NS) )
		{
			printf("BL: %2d, GSID: %2d, LSID: %2d, UID: %2d, F2: %+6.3e, CPS: %+6.3e, CPF: %+6.3e \n", \
			    BL, GlobalSystemID, LocalSystemID, UnitID, \
				r_NextState[BL][1], \
				s_CouplingStrength[LocalSystemID][CouplingIndex], r_CouplingFactor[BL][CouplingIndex]);
		}*/
		
		// Specific
		if ( (GlobalSystemID==GSID) && (UnitID==UID) && (LocalSystemID<SPB) )
		{
			printf("BL: %2d, GSID: %2d, LSID: %2d, UID: %2d, F2: %+6.3e, CPS: %+6.3e, CPF: %+6.3e \n", \
			    BL, GlobalSystemID, LocalSystemID, UnitID, \
				r_NextState[BL][1], \
				s_CouplingStrength[LocalSystemID][CouplingIndex], r_CouplingFactor[BL][CouplingIndex]);
		}
	}
	__syncthreads(); // -------------------------------------------------------
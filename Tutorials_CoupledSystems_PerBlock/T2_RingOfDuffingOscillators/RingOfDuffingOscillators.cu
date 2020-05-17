#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <random>

#include "RingOfDuffingOscillators_SystemDefinition.cuh"
#include "CoupledSystems_PerBlock_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Physical control parameters
const int NumberOfInitialConditionsX = 5; // Control parameter
const int NumberOfInitialConditionsY = 5; // Control parameter
const int NumberOfUnitsPerSystem = 14; // Number coupled units

// Solver Configuration
#define TIMESTEP   1.0e-2
#define TOLERANCE  1.0e-8
#define PERIOD     10.0

#define SOLVER RKCK45      // RK4, RKCK45
#define PRECISION double   // float, double
const int NS   = NumberOfInitialConditionsX * NumberOfInitialConditionsY; // NumberOfSystems
const int UPS  = NumberOfUnitsPerSystem;                // UnitsPerSystem
const int UD   = 2;     // UnitDimension
const int TPB  = 32;    // ThreadsPerBlock (integer multiple of the warp size that is 32)
const int SPB  = 1;     // SystemsPerBlock
const int NC   = 2;     // NumberOfCouplings
const int CBW  = 1;     // CouplingBandwidthRadius (0: full coupling matrix)
const int CCI  = 1;     // CouplingCircularity (0: non-circular matrix, 1: circular matrix)

const int NUP  = 3;     // NumberOfUnitParameters (different form system to system, different from unit to unit)
const int NSP  = 0;     // NumberOfSystemParameters (different from system to system, shared by all units)
const int NGP  = 1;     // NumberOfGlobalParameters (shared by all systems, share by all units)
const int NiGP = 0;     // NumberOfIntegerGlobalParameters (shared by all systems, shared by all units)

const int NUA  = 3;     // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 1;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 2;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 1;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 1;     // NumberOfEvents (per units)
const int NDO  = 500;     // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Random(vector<PRECISION>&, PRECISION, PRECISION, int, int);
void Gauss(vector<PRECISION>&, PRECISION, PRECISION, int);

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&);



int main()
{
	vector<PRECISION> ICX12(NumberOfInitialConditionsX,0);
	vector<PRECISION> ICY12(NumberOfInitialConditionsY,0);
	
	Linspace(ICX12,  -8.0,  8.0, NumberOfInitialConditionsX);
	Linspace(ICY12, -20.0, 40.0, NumberOfInitialConditionsY);
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION> ScanSystem(SelectedDevice);
	
	ScanSystem.SolverOption(InitialTimeStep, TIMESTEP);
	ScanSystem.SolverOption(RelativeTolerance, 0, TOLERANCE);
	ScanSystem.SolverOption(AbsoluteTolerance, 0, TOLERANCE);
	ScanSystem.SolverOption(EventDirection, 0, -1); // Detect local maxima
	ScanSystem.SolverOption(DenseOutputMinimumTimeStep, 1e-6);
	ScanSystem.SolverOption(DenseOutputSaveFrequency, 1);
	
	FillSolverObject(ScanSystem, ICX12, ICY12);
	FillCouplingMatrix(ScanSystem);
	
	
	
	ScanSystem.SynchroniseFromHostToDevice(All);
	ScanSystem.InsertSynchronisationPoint();
	ScanSystem.SynchroniseSolver();
	
	clock_t SimulationStart = clock();
	ScanSystem.Solve();
	ScanSystem.InsertSynchronisationPoint();
	ScanSystem.SynchroniseSolver();
	clock_t SimulationEnd = clock();
	
	ScanSystem.SynchroniseFromDeviceToHost(All);
	ScanSystem.InsertSynchronisationPoint();
	ScanSystem.SynchroniseSolver();
	
	
	
	ofstream DataFile;
	DataFile.open ( "Results.txt" );
	
	for (int sid=0; sid<NS; sid++)
	{
		DataFile << "SID: " << fixed << sid;
		DataFile << ", Ndt: " << ScanSystem.GetHost<int>(sid, IntegerSystemAccessories, 0);
		DataFile << scientific;
		DataFile << ", t: "   << ScanSystem.GetHost<PRECISION>(sid, ActualTime, 0);
		DataFile << ", dtmin: " << ScanSystem.GetHost<PRECISION>(sid, SystemAccessories, 0);
		DataFile << ", dtmax: " << ScanSystem.GetHost<PRECISION>(sid, SystemAccessories, 1) << endl;
		DataFile << " Initial conditions:";
		if ( UPS >= 12 )
		{
			DataFile << " X0_11: " << ScanSystem.GetHost<PRECISION>(sid, 11, UnitAccessories, 0);
			DataFile << " X1_11: " << ScanSystem.GetHost<PRECISION>(sid, 11, UnitAccessories, 1) << endl;
		}
		DataFile << "    States: " << endl;
		DataFile << "    UID,            X0,            X1," << endl;
		
		for (int uid=0; uid<UPS; uid++)
		{
			DataFile << "    " << setw(3) << fixed << uid << ", ";
			DataFile << scientific;
			DataFile << setw(13) << ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, 0) << ", ";
			DataFile << setw(13) << ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, 1) << ", " << endl;
		}
		
		DataFile << endl << endl;
	}
	
	DataFile << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	DataFile.close();
	
	for (int sid=0; sid<NS; sid++)
		ScanSystem.Print(DenseOutput, sid);
}

// ------------------------------------------------------------------------------------------------

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
	x[0]   = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}

void Logspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    x[0] = B; 
	
	if ( N>1 )
	{
		x[N-1] = E;
		PRECISION ExpB = log10(B);
		PRECISION ExpE = log10(E);
		PRECISION ExpIncr = (ExpE-ExpB)/(N-1);
		for (int i=1; i<N-1; i++)
		{
			x[i] = pow(10,ExpB + i*ExpIncr);
		}
	}
}

void Random(vector<PRECISION>& x, PRECISION B, PRECISION E, int N, int Res)
{
    srand(time(NULL));
	
	int Random;
	
	for (int i=0; i<N; i++)
	{
		Random = (rand() % Res);
		x[i] = B + Random*(E-B)/Res;
	}
}

void Gauss(vector<PRECISION>& x, PRECISION M, PRECISION D, int N)
{
    random_device rd;
	default_random_engine generator;
	generator.seed( rd() );
	normal_distribution<PRECISION> distribution(M,D);
	
	for (int i=0; i<N; i++)
	{
		distribution.reset();
		x[i] = distribution(generator);
	}
}

// ------------------------------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& ICX, const vector<PRECISION>& ICY)
{
	PRECISION CPS   = 0.004 / (2.0*(double)CBW);
	PRECISION GPAR  = 0.004;
	
	cout << "Coupling strength: " << CPS << endl;
	
	// Loop over the systems (blocks): control parameters/initial conditions
	int SystemNumber = 0;
	for (auto const& X0: ICX)
	{
		for (auto const& Y0: ICY)
		{			
			// SYSTEM SCOPE
			Solver.SetHost(SystemNumber, TimeDomain, 0, 0.0);
			Solver.SetHost(SystemNumber, TimeDomain, 1, PERIOD*2.0*PI/0.5);
			
			for (int i=0; i<NC; i++)
				Solver.SetHost(SystemNumber, CouplingStrength, i, CPS);
			
			Solver.SetHost(SystemNumber, SystemAccessories, 0, TIMESTEP);
			Solver.SetHost(SystemNumber, SystemAccessories, 1, TIMESTEP);
			
			// UNIT SCOPE
			int UnitNumber = 0;
			for (int i=0; i<UPS; i++) // Loop through units
			{
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, -5.0);
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 1, -5.0);
				
				Solver.SetHost(SystemNumber, UnitNumber, UnitAccessories, 0, -5.0); // Store initial conditions in an ACC
				Solver.SetHost(SystemNumber, UnitNumber, UnitAccessories, 1, -5.0); // Store initial conditions in an ACC
				
				if ( UnitNumber == 11 )
				{
					Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, X0);
					Solver.SetHost(SystemNumber, UnitNumber, ActualState, 1, Y0);
					
					Solver.SetHost(SystemNumber, UnitNumber, UnitAccessories, 0, X0); // Store initial conditions in an ACC
					Solver.SetHost(SystemNumber, UnitNumber, UnitAccessories, 1, Y0); // Store initial conditions in an ACC
				}
				
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  0, 0.5);
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  1, 0.24);
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  2, 13.633);
				
				UnitNumber++;
			}
			
			SystemNumber++;
		}
	}
	
	// GLOBAL SCOPE
	for (int i=0; i<NC; i++)
		Solver.SetHost(CouplingIndex, i, i);
	
	Solver.SetHost(GlobalParameters, 0, GPAR );
}

void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver)
{
	for (int Row=0; Row<UPS; Row++)
	{
		for (int Diag=-CBW; Diag<=CBW; Diag++)
		{
			int ModifiedCol = Row + Diag;
			
			if ( ModifiedCol >= UPS )
				ModifiedCol = ModifiedCol - UPS;
			
			if ( ModifiedCol < 0 )
				ModifiedCol = ModifiedCol + UPS;
			
			if ( Diag != 0 )
			{
				Solver.SetHost(0, CouplingMatrix, Row, ModifiedCol, 1.0);
				Solver.SetHost(1, CouplingMatrix, Row, ModifiedCol, 1.0);
			} else
			{
				Solver.SetHost(0, CouplingMatrix, Row, ModifiedCol, 0.0);
				Solver.SetHost(1, CouplingMatrix, Row, ModifiedCol, 0.0);
			}
		}
	}
}
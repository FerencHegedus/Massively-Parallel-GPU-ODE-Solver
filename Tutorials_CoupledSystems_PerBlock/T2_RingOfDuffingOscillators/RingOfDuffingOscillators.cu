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

#define COUPLINGSTRENGTH 1.0
#define DISTANCE 2000.0;

using namespace std;

// Physical control parameters
const int NumberOfInitialConditionsX = 5; // Control parameter
const int NumberOfInitialConditionsY = 5; // Control parameter
const int NumberOfUnitsPerSystem = 25; // Number coupled units

// Solver Configuration
#define TIMESTEP   0.5e-3
#define TOLERANCE  1.0e-10

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

const int NUA  = 2;     // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 1;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 3;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 1;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 1;     // NumberOfEvents (per units)
const int NDO  = 0;     // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Random(vector<PRECISION>&, PRECISION, PRECISION, int, int);
void Gauss(vector<PRECISION>&, PRECISION, PRECISION, int);

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&);

void SaveData(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, ofstream&, int, int, int);

// ------------------------------------------------------------------------------------------------

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
	
	ScanSystem.SolverOption(SharedCouplingMatrices, 0);
	ScanSystem.SolverOption(InitialTimeStep, TIMESTEP);
	ScanSystem.SolverOption(RelativeTolerance, 0, TOLERANCE);
	ScanSystem.SolverOption(AbsoluteTolerance, 0, TOLERANCE);
	ScanSystem.SolverOption(MinimumTimeStep, 1e-13);
	
	
	
	FillSolverObject(ScanSystem, ICX12, ICY12);
	FillCouplingMatrix(ScanSystem);
	
	ScanSystem.Print(TimeDomain);
	ScanSystem.Print(ActualState);
	ScanSystem.Print(UnitParameters);
	ScanSystem.Print(UnitAccessories);
	ScanSystem.Print(GlobalParameters);
	ScanSystem.Print(CouplingMatrix,0);
	ScanSystem.Print(CouplingMatrix,1);
	
	ofstream DataFile;
	DataFile.open ( "OrderNumber_200_1.txt" );
	ScanSystem.SynchroniseFromHostToDevice(All);
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	for (int i=0; i<1; i++)
	{
		ScanSystem.Solve();
		ScanSystem.SynchroniseFromDeviceToHost(All);
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		//SaveData(ScanSystem, DataFile, NS, UPS, i);
	}
	clock_t SimulationEnd = clock();
	
	
	int Width = 18;
	DataFile.precision(16);
	DataFile.flags(ios::scientific);
	for (int i=0; i<NS; i++)
	{
		PRECISION OrderParameter;
		PRECISION Error;
		
		int Sum=0;
		for (int j=0; j<UPS; j++)
		{
			PRECISION Xi   = ScanSystem.GetHost<PRECISION>(i, j, ActualState, 0);
			PRECISION Xip1 = ScanSystem.GetHost<PRECISION>(i, (j+1 < UPS ? j+1 : 0), ActualState, 0);
			
			Error = abs( Xi-Xip1 );
			Sum   = Sum + ( Error > 0.01 ? 1 : 0);
		}
		OrderParameter = (PRECISION)Sum / (PRECISION)UPS;
		
		//cout << "System number: " << i << " Order parameter: " << OrderParameter << endl;
		
		DataFile.width(Width); DataFile << i << ',';
		DataFile.width(Width); DataFile << ScanSystem.GetHost<PRECISION>(i, 11, UnitAccessories, 0) << ',';
		DataFile.width(Width); DataFile << ScanSystem.GetHost<PRECISION>(i, 11, UnitAccessories, 1) << ',';
		DataFile.width(Width); DataFile << OrderParameter << '\n';
	}
	
	
	/*int Width = 18;
	DataFile.precision(16);
	DataFile.flags(ios::scientific);
	DataFile.width(Width); DataFile << ScanSystem.GetHost<PRECISION>(0, 0, ActualState, 0) << '\n';
	DataFile.width(Width); DataFile << ScanSystem.GetHost<PRECISION>(0, SystemAccessories, 0) << '\n';
	DataFile.width(Width); DataFile << ScanSystem.GetHost<PRECISION>(0, SystemAccessories, 1) << '\n';
	DataFile.width(Width); DataFile << ScanSystem.GetHost<PRECISION>(0, SystemAccessories, 2) << '\n';*/
	
	
	DataFile.close();
	
	/*for (int sid=0; sid<NS; sid++)
	{
		for (int uid=0; uid<UPS; uid++)
		{
			cout << "SID: " << sid << ", UID: " << uid << ", uACC[0]: " << ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, 0) \
			                                           << ", uACC[1]: " << ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, 1) << endl;
		}
	}*/
	
	for (int sid=0; sid<NS; sid++)
	{
		cout << "SID: " << sid << ", number of time steps: " << ScanSystem.GetHost<PRECISION>(sid, IntegerSystemAccessories, 0);
		cout << ", time instant: " << ScanSystem.GetHost<PRECISION>(sid, ActualTime, 0);
		cout << ", X00: " << ScanSystem.GetHost<PRECISION>(sid, 11, ActualState, 0);
		cout << ", X10: " << ScanSystem.GetHost<PRECISION>(sid, 11, ActualState, 1) << endl;
	}
	
	
	cout <<  "Maximum time step: " << ScanSystem.GetHost<PRECISION>(0, SystemAccessories, 0) << endl;
	cout <<  "Minimum time step: " << ScanSystem.GetHost<PRECISION>(0, SystemAccessories, 1) << endl;
	
	cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	ScanSystem.Print(ActualTime);
	
	// SIMULATION -------------------------------------------------------------
	
	/*ofstream DataFile;
	
	string       Filename;
	stringstream StreamFilename;
	
	PRECISION Tmp1 = DISTANCE;
	PRECISION Tmp2 = COUPLINGSTRENGTH;
	StreamFilename.str("");
	StreamFilename.precision(0);
	StreamFilename.setf(ios::fixed);
	//StreamFilename << "BubbleEnsemble_" << Frequency[0] << "_" << Tmp1 << "_" << Tmp2 << ".txt";
	StreamFilename << "TestCase_N64_1.txt";
	Filename = StreamFilename.str();
		
	DataFile.open ( Filename.c_str() );
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	ScanSystem.SynchroniseFromHostToDevice(All);
	// Transients
	for (int i=0; i<100; i++)
	{
		ScanSystem.Solve();
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		//cout << "Transient finished: " << i << endl;
	}
	
		ScanSystem.SynchroniseFromDeviceToHost(All);
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
		SaveData(ScanSystem, DataFile, NS, UPS, UD);
		
	// Save Poincare section
	/*for (int i=0; i<0; i++)
	{
		ScanSystem.Solve();
		ScanSystem.SynchroniseFromDeviceToHost(All);
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		SaveData(ScanSystem, DataFile, NS, UPS, UD);
		
		cout << "Poincare finished: " << i << endl;
	}*/
	
	
	
	//DataFile.close();
	
	//cout << "SIMULATION COMPLETED!" << endl;
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
			Solver.SetHost(SystemNumber, TimeDomain, 1, 0.01*2.0*PI/0.5);
			
			for (int i=0; i<NC; i++)
				Solver.SetHost(SystemNumber, CouplingStrength, i, CPS);
			
			Solver.SetHost(SystemNumber, SystemAccessories, 0, TIMESTEP);
			Solver.SetHost(SystemNumber, SystemAccessories, 1, TIMESTEP);
			Solver.SetHost(SystemNumber, SystemAccessories, 2, 0);
			
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


// ------------------------------------------------------------------------------------------------

void SaveData(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, ofstream& DataFile, int NS, int UPS, int trn)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int sid=0; sid<NS; sid++)
	{
		DataFile.width(4); DataFile << trn << ',';
		//DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, 0, UnitParameters, 0) << ',';
		for (int uid=0; uid<UPS; uid++)
		{
			DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, uid, ActualState, 0) << ',';
		}
		DataFile << '\n';
	}
}
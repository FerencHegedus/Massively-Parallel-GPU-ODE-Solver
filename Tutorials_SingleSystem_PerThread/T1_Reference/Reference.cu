#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "Reference_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Solver Configuration
#define SOLVER RKCK45     // RK4, RKCK45
#define PRECISION double  // float, double
const int NT   = 23040; // NumberOfThreads
const int SD   = 2;     // SystemDimension
const int NCP  = 1;     // NumberOfControlParameters
const int NSP  = 1;     // NumberOfSharedParameters
const int NISP = 0;     // NumberOfIntegerSharedParameters
const int NE   = 2;     // NumberOfEvents
const int NA   = 3;     // NumberOfAccessories
const int NIA  = 1;     // NumberOfIntegerAccessories
const int NDO  = 200;   // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, PRECISION, PRECISION, PRECISION, int, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, ofstream&, int);

int main()
{
	int NumberOfProblems = 46080; // 2*NT;
	int BlockSize        = 64;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	PRECISION InitialConditions_X1 = -0.5;
	PRECISION InitialConditions_X2 = -0.1;
	PRECISION Parameters_B = 0.3;
	
	int NumberOfParameters_k = NumberOfProblems;
	PRECISION kRangeLower = 0.2;
    PRECISION kRangeUpper = 0.3;
		vector<PRECISION> Parameters_k_Values(NumberOfParameters_k,0);
		Linspace(Parameters_k_Values, kRangeLower, kRangeUpper, NumberOfParameters_k);
	
	
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanDuffing(SelectedDevice);
	
	ScanDuffing.SolverOption(PreferSharedMemory, 0);
	ScanDuffing.SolverOption(ThreadsPerBlock, BlockSize);
	ScanDuffing.SolverOption(InitialTimeStep, 1.0e-2);
	ScanDuffing.SolverOption(ActiveNumberOfThreads, NT);
	
	ScanDuffing.SolverOption(DenseOutputMinimumTimeStep, 0.0);
	ScanDuffing.SolverOption(DenseOutputSaveFrequency, 1);
	
	ScanDuffing.SolverOption(MaximumTimeStep, 1.0e3);
	ScanDuffing.SolverOption(MinimumTimeStep, 1.0e-14);
	ScanDuffing.SolverOption(TimeStepGrowLimit, 10.0);
	ScanDuffing.SolverOption(TimeStepShrinkLimit, 0.2);
	
	ScanDuffing.SolverOption(RelativeTolerance, 0, 1e-9);
	ScanDuffing.SolverOption(RelativeTolerance, 1, 1e-9);
	ScanDuffing.SolverOption(AbsoluteTolerance, 0, 1e-9);
	ScanDuffing.SolverOption(AbsoluteTolerance, 1, 1e-9);
	
	ScanDuffing.SolverOption(EventTolerance, 0, 1e-6);
	ScanDuffing.SolverOption(EventTolerance, 1, 1e-6);
	ScanDuffing.SolverOption(EventDirection, 0, -1);
	ScanDuffing.SolverOption(EventDirection, 1,  0);
	
	
	int NumberOfSimulationLaunches = NumberOfProblems / NT + (NumberOfProblems % NT == 0 ? 0:1);
	
	ofstream DataFile;
	DataFile.open ( "Duffing.txt" );
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		FillSolverObject(ScanDuffing, Parameters_k_Values, Parameters_B, InitialConditions_X1, InitialConditions_X2, LaunchCounter * NT, NT);
	
		ScanDuffing.SynchroniseFromHostToDevice(All);
		ScanDuffing.InsertSynchronisationPoint();
		ScanDuffing.SynchroniseSolver();
		
		TransientStart = clock();
		for (int i=0; i<1024; i++)
		{
			ScanDuffing.Solve();
			ScanDuffing.InsertSynchronisationPoint();
			ScanDuffing.SynchroniseSolver();
		}
		TransientEnd = clock();
			cout << "Launches: " << LaunchCounter << "  Simulation time: " << 1000.0*(TransientEnd-TransientStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
		
		for (int i=0; i<32; i++)
		{
			ScanDuffing.Solve();
			ScanDuffing.SynchroniseFromDeviceToHost(All);
			ScanDuffing.InsertSynchronisationPoint();
			ScanDuffing.SynchroniseSolver();
			
			SaveData(ScanDuffing, DataFile, NT);
		}
	}
	
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	DataFile.close();
	
	ScanDuffing.Print(DenseOutput, 0);
	ScanDuffing.Print(DenseOutput, 4789);
	ScanDuffing.Print(DenseOutput, 15479);
	
	cout << "Test finished!" << endl;
}

// AUXILIARY FUNCTION -----------------------------------------------------------------------------

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

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& k_Values, PRECISION B, PRECISION X10, PRECISION X20, int FirstProblemNumber, int NumberOfThreads)
{
	int k_begin = FirstProblemNumber;
	int k_end   = FirstProblemNumber + NumberOfThreads;
	
	int ProblemNumber = 0;
	for (int k=k_begin; k<k_end; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0.0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, 2*PI );
		
		Solver.SetHost(ProblemNumber, ActualState, 0, X10 );
		Solver.SetHost(ProblemNumber, ActualState, 1, X20 );
		
		Solver.SetHost(ProblemNumber, ActualTime, 0.0 );
		
		Solver.SetHost(ProblemNumber, ControlParameters, 0, k_Values[k] );
		
		Solver.SetHost(ProblemNumber, Accessories, 0, 0.0 );
		Solver.SetHost(ProblemNumber, Accessories, 1, 0.0 );
		Solver.SetHost(ProblemNumber, Accessories, 2, 0.0 );
		
		Solver.SetHost(ProblemNumber, DenseIndex, 0 );
		
		ProblemNumber++;
	}
	
	Solver.SetHost(SharedParameters, 0, B );
}

void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, ofstream& DataFile, int NumberOfThreads)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(SharedParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, Accessories, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, Accessories, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, Accessories, 2);
		DataFile << '\n';
	}
}
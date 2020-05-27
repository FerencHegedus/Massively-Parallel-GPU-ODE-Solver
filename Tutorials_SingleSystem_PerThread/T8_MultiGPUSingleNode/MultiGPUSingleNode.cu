#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "MultiGPUSingleNode_SystemDefinition.cuh"
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
void FillSolverObjects(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, PRECISION, PRECISION, PRECISION, int, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, ofstream&, int);


int main()
{
	int NumberOfProblems = NT*2;
	int NumberOfThreads  = NT;
	int BlockSize        = 64;
	
	
	ListCUDADevices();
	int SelectedDevice1 = 0; // According to the output of the function call ListCUDADevices();
	int SelectedDevice2 = 2; // THEY MUST BE SET ACCORDING TO YOUR CURRENT CONFIGURATION!!!
	
	
	PRECISION InitialConditions_X1 = -0.5;
	PRECISION InitialConditions_X2 = -0.1;
	PRECISION Parameters_B = 0.3;
	
	int NumberOfParameters_k = NumberOfProblems;
	PRECISION kRangeLower = 0.2;
    PRECISION kRangeUpper = 0.3;
		vector<PRECISION> Parameters_k_Values(NumberOfParameters_k,0);
		Linspace(Parameters_k_Values, kRangeLower, kRangeUpper, NumberOfParameters_k);
	
	
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanDuffing1(SelectedDevice1);
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanDuffing2(SelectedDevice2);
	
	ScanDuffing1.SolverOption(ThreadsPerBlock, BlockSize);
	ScanDuffing2.SolverOption(ThreadsPerBlock, BlockSize);
	
	ScanDuffing1.SolverOption(RelativeTolerance, 0, 1.0e-9);
	ScanDuffing1.SolverOption(RelativeTolerance, 1, 1.0e-9);
	ScanDuffing1.SolverOption(AbsoluteTolerance, 0, 1.0e-9);
	ScanDuffing1.SolverOption(AbsoluteTolerance, 1, 1.0e-9);
	
	ScanDuffing2.SolverOption(RelativeTolerance, 0, 1.0e-9);
	ScanDuffing2.SolverOption(RelativeTolerance, 1, 1.0e-9);
	ScanDuffing2.SolverOption(AbsoluteTolerance, 0, 1.0e-9);
	ScanDuffing2.SolverOption(AbsoluteTolerance, 1, 1.0e-9);
	
	ScanDuffing1.SolverOption(EventDirection, 0, -1);
	ScanDuffing2.SolverOption(EventDirection, 0, -1);
	
	
	int NumberOfSimulationLaunches = NumberOfProblems / NumberOfThreads / 2;
	
	ofstream DataFile;
	DataFile.open ( "Duffing.txt" );
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	int FirstProblemNumber;
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		FirstProblemNumber = LaunchCounter * (2*NumberOfThreads);
		
		FillSolverObjects(ScanDuffing1, Parameters_k_Values, Parameters_B, InitialConditions_X1, InitialConditions_X2, FirstProblemNumber, NumberOfThreads);
		ScanDuffing1.SynchroniseFromHostToDevice(All);
		ScanDuffing1.Solve();
		ScanDuffing1.InsertSynchronisationPoint();
		
		FirstProblemNumber = FirstProblemNumber + NumberOfThreads;
		
		FillSolverObjects(ScanDuffing2, Parameters_k_Values, Parameters_B, InitialConditions_X1, InitialConditions_X2, FirstProblemNumber, NumberOfThreads);
		ScanDuffing2.SynchroniseFromHostToDevice(All);
		ScanDuffing2.Solve();
		ScanDuffing2.InsertSynchronisationPoint();
		
		TransientStart = clock();
		for (int i=0; i<1023; i++)
		{
			ScanDuffing1.SynchroniseSolver();
			ScanDuffing1.Solve();
			ScanDuffing1.InsertSynchronisationPoint();
			
			ScanDuffing2.SynchroniseSolver();
			ScanDuffing2.Solve();
			ScanDuffing2.InsertSynchronisationPoint();
		}
		TransientEnd = clock();
			cout << "Transient iteration: " << LaunchCounter << "  Simulation time: " << 1000.0*(TransientEnd-TransientStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
		
		
		ScanDuffing1.SynchroniseSolver();
		ScanDuffing2.SynchroniseSolver();
		for (int i=0; i<32; i++)
		{
			ScanDuffing1.Solve();
			ScanDuffing1.SynchroniseFromDeviceToHost(All);
			ScanDuffing1.InsertSynchronisationPoint();
			
			ScanDuffing2.Solve();
			ScanDuffing2.SynchroniseFromDeviceToHost(All);
			ScanDuffing2.InsertSynchronisationPoint();
			
			
			ScanDuffing1.SynchroniseSolver();
			SaveData(ScanDuffing1, DataFile, NumberOfThreads);
			
			ScanDuffing2.SynchroniseSolver();
			SaveData(ScanDuffing2, DataFile, NumberOfThreads);
		}
	}
	
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	
	DataFile.close();
	
	cout << "Test finished!" << endl;
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

// ------------------------------------------------------------------------------------------------

void FillSolverObjects(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& k_Values, PRECISION B, PRECISION X10, PRECISION X20, int FirstProblemNumber, int NumberOfThreads)
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
		
		Solver.SetHost(ProblemNumber, ControlParameters, 0, k_Values[k] );
		
		Solver.SetHost(ProblemNumber, Accessories, 0, 0.0 );
		Solver.SetHost(ProblemNumber, Accessories, 1, 0.0 );
		Solver.SetHost(ProblemNumber, Accessories, 2, 0.0 );
		
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
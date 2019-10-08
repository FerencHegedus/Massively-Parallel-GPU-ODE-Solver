#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "SingleSystem_PerThread_IndexingMacroEnabled.cuh"
#include "DoubleBuffering_SystemDefinition.cuh"
#include "SingleSystem_PerThread_IndexingMacroDisabled.cuh"
#include "SingleSystem_PerThread.cuh"

#define PI 3.14159265358979323846

using namespace std;

void Linspace(vector<double>&, double, double, int);
void FillSolverObjects(ProblemSolver&, const vector<double>&, double, double, double, int, int);
void SaveData(ProblemSolver&, ofstream&, int);


int main()
{
	int NumberOfProblems = 46080;
	int NumberOfThreads  = 23040;
	int BlockSize        = 64;
	
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	double InitialConditions_X1 = -0.5;
	double InitialConditions_X2 = -0.1;
	double Parameters_B = 0.3;
	
	int NumberOfParameters_k = NumberOfProblems;
	double kRangeLower = 0.2;
    double kRangeUpper = 0.3;
		vector<double> Parameters_k_Values(NumberOfParameters_k,0);
		Linspace(Parameters_k_Values, kRangeLower, kRangeUpper, NumberOfParameters_k);
	
	
	ConstructorConfiguration ConfigurationDuffing;
	
	ConfigurationDuffing.NumberOfThreads           = NumberOfThreads;
	ConfigurationDuffing.SystemDimension           = 2;
	ConfigurationDuffing.NumberOfControlParameters = 1;
	ConfigurationDuffing.NumberOfSharedParameters  = 1;
	ConfigurationDuffing.NumberOfEvents            = 2;
	ConfigurationDuffing.NumberOfAccessories       = 3;
	ConfigurationDuffing.DenseOutputNumberOfPoints = 0;
	
	ProblemSolver ScanDuffing1(ConfigurationDuffing, SelectedDevice);
	ProblemSolver ScanDuffing2(ConfigurationDuffing, SelectedDevice);
	
	ScanDuffing1.SolverOption(ThreadsPerBlock, BlockSize);
	ScanDuffing2.SolverOption(ThreadsPerBlock, BlockSize);
	
	ScanDuffing1.SolverOption(RelativeTolerance, 0, 1e-9);
	ScanDuffing1.SolverOption(RelativeTolerance, 1, 1e-9);
	ScanDuffing1.SolverOption(AbsoluteTolerance, 0, 1e-9);
	ScanDuffing1.SolverOption(AbsoluteTolerance, 1, 1e-9);
	
	ScanDuffing2.SolverOption(RelativeTolerance, 0, 1e-9);
	ScanDuffing2.SolverOption(RelativeTolerance, 1, 1e-9);
	ScanDuffing2.SolverOption(AbsoluteTolerance, 0, 1e-9);
	ScanDuffing2.SolverOption(AbsoluteTolerance, 1, 1e-9);
	
	ScanDuffing1.SolverOption(EventDirection, 0, -1);
	ScanDuffing2.SolverOption(EventDirection, 0, -1);
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
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
		for (int i=0; i<1024; i++)
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
		
		for (int i=0; i<31; i++)
		{
			ScanDuffing1.SynchroniseSolver();
			ScanDuffing1.SynchroniseFromDeviceToHost(All);
			ScanDuffing1.Solve();
			ScanDuffing1.InsertSynchronisationPoint();
			
			SaveData(ScanDuffing1, DataFile, NumberOfThreads);
			
			
			ScanDuffing2.SynchroniseSolver();
			ScanDuffing2.SynchroniseFromDeviceToHost(All);
			ScanDuffing2.Solve();
			ScanDuffing2.InsertSynchronisationPoint();
			
			SaveData(ScanDuffing2, DataFile, NumberOfThreads);
		}
		
		ScanDuffing1.SynchroniseSolver();
		ScanDuffing1.SynchroniseFromDeviceToHost(All);
		SaveData(ScanDuffing1, DataFile, NumberOfThreads);
		
		ScanDuffing2.SynchroniseSolver();
		ScanDuffing2.SynchroniseFromDeviceToHost(All);
		SaveData(ScanDuffing2, DataFile, NumberOfThreads);
	}
	
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	
	DataFile.close();
	
	cout << "Test finished!" << endl;
}

// ------------------------------------------------------------------------------------------------

void Linspace(vector<double>& x, double B, double E, int N)
{
    double Increment;
	
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

void FillSolverObjects(ProblemSolver& Solver, const vector<double>& k_Values, double B, double X10, double X20, int FirstProblemNumber, int NumberOfThreads)
{
	int k_begin = FirstProblemNumber;
	int k_end   = FirstProblemNumber + NumberOfThreads;
	
	int ProblemNumber = 0;
	for (int k=k_begin; k<k_end; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, 2*PI );
		
		Solver.SetHost(ProblemNumber, ActualState, 0, X10 );
		Solver.SetHost(ProblemNumber, ActualState, 1, X20 );
		
		Solver.SetHost(ProblemNumber, ControlParameters, 0, k_Values[k] );
		
		Solver.SetHost(ProblemNumber, Accessories, 0, 0 );
		Solver.SetHost(ProblemNumber, Accessories, 1, 0 );
		Solver.SetHost(ProblemNumber, Accessories, 2, 0 );
		
		ProblemNumber++;
	}
	
	Solver.SetHost(SharedParameters, 0, B );
}

void SaveData(ProblemSolver& Solver, ofstream& DataFile, int NumberOfThreads)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.GetHost(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost(SharedParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost(tid, ActualState, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost(tid, Accessories, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost(tid, Accessories, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost(tid, Accessories, 2);
		DataFile << '\n';
	}
}
/*
Third tutorial example: T3 (Double Buffering of the reference simulation)
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "MassivelyParallel_GPU-ODE_Solver.cuh"

#define PI 3.14159265358979323846

using namespace std;

void Linspace(vector<double>&, double, double, int);
void FillControlParameters(ProblemSolver&, const vector<double>&, double, double, int, int);
void FillSharedParameters(ProblemSolver&, double);
void SaveData(ProblemSolver&, ofstream&, int);


int main()
{
	int PoolSize        = 46080;
	int NumberOfThreads = 23040;
	int BlockSize       = 64;
	
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	double InitialConditions_X1 = -0.5;
	double InitialConditions_X2 = -0.1;
	double Parameters_B = 0.3;
	
	int NumberOfParameters_k = PoolSize;
	double kRangeLower = 0.2;
    double kRangeUpper = 0.3;
		vector<double> Parameters_k_Values(NumberOfParameters_k,0);
		Linspace(Parameters_k_Values, kRangeLower, kRangeUpper, NumberOfParameters_k);
	
	
	ConstructorConfiguration ConfigurationDuffing;
		ConfigurationDuffing.PoolSize                  = PoolSize;
		ConfigurationDuffing.NumberOfThreads           = NumberOfThreads;
		ConfigurationDuffing.SystemDimension           = 2;
		ConfigurationDuffing.NumberOfControlParameters = 1;
		ConfigurationDuffing.NumberOfSharedParameters  = 1;
		ConfigurationDuffing.NumberOfEvents            = 2;
		ConfigurationDuffing.NumberOfAccessories       = 4;
	
	
	ProblemSolver ScanDuffing1(ConfigurationDuffing, SelectedDevice);
	ProblemSolver ScanDuffing2(ConfigurationDuffing, SelectedDevice);
	
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
	int NumberOfSimulationLaunches = PoolSize / NumberOfThreads / 2;
	
	
	SolverConfiguration SolverConfigurationSystem;
	
	SolverConfigurationSystem.BlockSize       = BlockSize;
	SolverConfigurationSystem.InitialTimeStep = 1e-2;
	SolverConfigurationSystem.Solver          = RKCK45;
	SolverConfigurationSystem.ActiveThreads   = NumberOfThreads;
	
	
	ofstream DataFile;
	DataFile.open ( "Duffing.txt" );
	
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	
	FillSharedParameters(ScanDuffing1,Parameters_B);
	ScanDuffing1.SynchroniseSharedFromHostToDeviceAsync();
	
	FillSharedParameters(ScanDuffing2,Parameters_B);
	ScanDuffing2.SynchroniseSharedFromHostToDeviceAsync();
	
	
	int FirstProblemNumber;
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		FirstProblemNumber = LaunchCounter * (2*NumberOfThreads);
		
		FillControlParameters(ScanDuffing1, Parameters_k_Values, InitialConditions_X1, InitialConditions_X2, FirstProblemNumber, NumberOfThreads);
		ScanDuffing1.SynchroniseFromHostToDeviceAsync(All);
		ScanDuffing1.SolveAsync(SolverConfigurationSystem);
		ScanDuffing1.InsertSynchronisationPoint();
		
		FirstProblemNumber = FirstProblemNumber + NumberOfThreads;
		
		FillControlParameters(ScanDuffing2, Parameters_k_Values, InitialConditions_X1, InitialConditions_X2, FirstProblemNumber, NumberOfThreads);
		ScanDuffing2.SynchroniseFromHostToDeviceAsync(All);
		ScanDuffing2.SolveAsync(SolverConfigurationSystem);
		ScanDuffing2.InsertSynchronisationPoint();
		
		TransientStart = clock();
		for (int i=0; i<1024; i++)
		{
			ScanDuffing1.SynchroniseSolver();
			ScanDuffing1.SolveAsync(SolverConfigurationSystem);
			ScanDuffing1.InsertSynchronisationPoint();
			
			ScanDuffing2.SynchroniseSolver();
			ScanDuffing2.SolveAsync(SolverConfigurationSystem);
			ScanDuffing2.InsertSynchronisationPoint();
		}
		TransientEnd = clock();
			cout << "Transient iteration: " << LaunchCounter << "  Simulation time: " << 1000.0*(TransientEnd-TransientStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
		
		for (int i=0; i<31; i++)
		{
			ScanDuffing1.SynchroniseSolver();
			ScanDuffing1.SynchroniseFromDeviceToHostAsync(All);
			ScanDuffing1.SolveAsync(SolverConfigurationSystem);
			ScanDuffing1.InsertSynchronisationPoint();
			
			SaveData(ScanDuffing1, DataFile, NumberOfThreads);
			
			
			ScanDuffing2.SynchroniseSolver();
			ScanDuffing2.SynchroniseFromDeviceToHostAsync(All);
			ScanDuffing2.SolveAsync(SolverConfigurationSystem);
			ScanDuffing2.InsertSynchronisationPoint();
			
			SaveData(ScanDuffing2, DataFile, NumberOfThreads);
		}
		
		ScanDuffing1.SynchroniseSolver();
		ScanDuffing1.SynchroniseFromDeviceToHostAsync(All);
		SaveData(ScanDuffing1, DataFile, NumberOfThreads);
		
		ScanDuffing2.SynchroniseSolver();
		ScanDuffing2.SynchroniseFromDeviceToHostAsync(All);
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

void FillControlParameters(ProblemSolver& Solver, const vector<double>& k_Values, double X10, double X20, int FirstProblemNumber, int NumberOfThreads)
{
	int k_begin = FirstProblemNumber;
	int k_end   = FirstProblemNumber + NumberOfThreads;
	
	int ProblemNumber = 0;
	for (int k=k_begin; k<k_end; k++)
	{
		Solver.SingleSetHost(ProblemNumber, TimeDomain,  0, 0 );
		Solver.SingleSetHost(ProblemNumber, TimeDomain,  1, 2*PI );
		
		Solver.SingleSetHost(ProblemNumber, ActualState, 0, X10 );
		Solver.SingleSetHost(ProblemNumber, ActualState, 1, X20 );
		
		Solver.SingleSetHost(ProblemNumber, ControlParameters, 0, k_Values[k] );
		
		Solver.SingleSetHost(ProblemNumber, Accessories, 0, 0 );
		Solver.SingleSetHost(ProblemNumber, Accessories, 1, 0 );
		Solver.SingleSetHost(ProblemNumber, Accessories, 2, 0 );
		Solver.SingleSetHost(ProblemNumber, Accessories, 3, 1e-2 );
		
		ProblemNumber++;
	}
}

void SaveData(ProblemSolver& Solver, ofstream& DataFile, int NumberOfThreads)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.SharedGetHost(0) << ',';
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, ActualState, 1) << ',';
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, Accessories, 0) << ',';
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, Accessories, 1) << ',';
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, Accessories, 2) << ',';
		DataFile.width(Width); DataFile << Solver.SingleGetHost(tid, Accessories, 3) << ',';
		DataFile << '\n';
	}
}

void FillSharedParameters(ProblemSolver& Solver, double B)
{
	Solver.SetSharedHost(0, B );
}
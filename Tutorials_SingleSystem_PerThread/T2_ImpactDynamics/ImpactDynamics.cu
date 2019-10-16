#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "SingleSystem_PerThread_IndexingMacroEnabled.cuh"
#include "ImpactDynamics_SystemDefinition.cuh"
#include "SingleSystem_PerThread_IndexingMacroDisabled.cuh"
#include "SingleSystem_PerThread.cuh"

#define PI 3.14159265358979323846

#define SOLVER RKCK45
#define EVNT   EVNT1
#define DOUT   DOUT0

using namespace std;

void Linspace(vector<double>&, double, double, int);
void Logspace(vector<double>&, double, double, int);

void FillSolverObject(ProblemSolver<SOLVER,EVNT,DOUT>&, const vector<double>&);

int main()
{
	// The control parameter
	int NumberOfFlowRates = 30720;
	
	vector<double> FlowRates(NumberOfFlowRates,0);
	Linspace(FlowRates, 0.2, 10.0, NumberOfFlowRates);
	
	
	// Setup CUDA a device
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	// Problem Pool and Solver Object configuration
	int NumberOfProblems = NumberOfFlowRates; // 30720
	int NumberOfThreads  = NumberOfFlowRates; // 30720 -> 1 launches
	
	ConstructorConfiguration ConfigurationPressureReliefValve;
	
	ConfigurationPressureReliefValve.NumberOfThreads           = NumberOfThreads;
	ConfigurationPressureReliefValve.SystemDimension           = 3;
	ConfigurationPressureReliefValve.NumberOfControlParameters = 1;
	ConfigurationPressureReliefValve.NumberOfSharedParameters  = 4;
	ConfigurationPressureReliefValve.NumberOfEvents            = 2;
	ConfigurationPressureReliefValve.NumberOfAccessories       = 2;
	
	ProblemSolver<SOLVER,EVNT,DOUT> ScanPressureReliefValve(ConfigurationPressureReliefValve, SelectedDevice);
	
	ScanPressureReliefValve.SolverOption(ThreadsPerBlock, 64);
	ScanPressureReliefValve.SolverOption(RelativeTolerance, 0, 1e-10);
	ScanPressureReliefValve.SolverOption(RelativeTolerance, 1, 1e-10);
	ScanPressureReliefValve.SolverOption(RelativeTolerance, 2, 1e-10);
	ScanPressureReliefValve.SolverOption(AbsoluteTolerance, 0, 1e-10);
	ScanPressureReliefValve.SolverOption(AbsoluteTolerance, 1, 1e-10);
	ScanPressureReliefValve.SolverOption(AbsoluteTolerance, 2, 1e-10);
	ScanPressureReliefValve.SolverOption(EventDirection,   0, -1);
	ScanPressureReliefValve.SolverOption(EventDirection,   1, -1);
	ScanPressureReliefValve.SolverOption(EventStopCounter, 0,  1);
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
	int NumberOfSimulationLaunches = NumberOfProblems / NumberOfThreads;
	int ProblemStartIndex;
	
	ofstream DataFile;
	DataFile.open ( "PressureReliefValve.txt" );
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);
	
	
	clock_t SimulationStart = clock();
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		// Fill Solver Object
		ProblemStartIndex = LaunchCounter * NumberOfThreads;
		FillSolverObject(ScanPressureReliefValve, FlowRates); // There is only 1 launch; thus, no special care is needed!
		ScanPressureReliefValve.SynchroniseFromHostToDevice(All);
		
		
		// Transient simulations
		for (int i=0; i<1024; i++)
		{
			ScanPressureReliefValve.Solve();
			ScanPressureReliefValve.InsertSynchronisationPoint();
			ScanPressureReliefValve.SynchroniseSolver();
		}
		
		
		// Converged simulations and their data collection
		for (int i=0; i<32; i++)
		{
			ScanPressureReliefValve.Solve();
			ScanPressureReliefValve.SynchroniseFromDeviceToHost(Accessories);
			ScanPressureReliefValve.InsertSynchronisationPoint();
			ScanPressureReliefValve.SynchroniseSolver();
			
			for (int tid=0; tid<NumberOfThreads; tid++)
			{
				DataFile.width(Width); DataFile << ScanPressureReliefValve.GetHost(tid, ControlParameters, 0) << ',';
				DataFile.width(Width); DataFile << ScanPressureReliefValve.GetHost(tid, Accessories, 0) << ',';
				DataFile.width(Width); DataFile << ScanPressureReliefValve.GetHost(tid, Accessories, 1) << ',';
				DataFile << '\n';
			}
		}
	}
	
	DataFile.close();
	
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
}

// ------------------------------------------------------------------------------------------------

void Linspace(vector<double>& x, double B, double E, int N)
{
    double Increment;
	
	x[0] = B;
	
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

void Logspace(vector<double>& x, double B, double E, int N)
{
    x[0] = B; 
	
	if ( N>1 )
	{
		x[N-1] = E;
		double ExpB = log10(B);
		double ExpE = log10(E);
		double ExpIncr = (ExpE-ExpB)/(N-1);
		for (int i=1; i<N-1; i++)
		{
			x[i] = pow(10,ExpB + i*ExpIncr);
		}
	}
}

// ------------------------------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<SOLVER,EVNT,DOUT>& Solver, const vector<double>& q_Values)
{	
	int ProblemNumber = 0;
	for (auto const& q: q_Values) // dimensionless flow rate [-]
	{
		Solver.SetHost(ProblemNumber, TimeDomain, 0, 0);
		Solver.SetHost(ProblemNumber, TimeDomain, 1, 1e10); // Stopped by Poincaré section
		
		Solver.SetHost(ProblemNumber, ActualState, 0, 0.2);
		Solver.SetHost(ProblemNumber, ActualState, 1, 0.0);
		Solver.SetHost(ProblemNumber, ActualState, 2, 0.0);
		
		Solver.SetHost(ProblemNumber, ControlParameters,  0, q);
		
		ProblemNumber++;
	}
	
	Solver.SetHost(SharedParameters, 0, 1.25 ); // Kappa: damping coefficient       [-]
	Solver.SetHost(SharedParameters, 1, 10.0 ); // Delta: spring precompression     [-]
	Solver.SetHost(SharedParameters, 2, 20.0 ); // Beta:  compressibility parameter [-]
	Solver.SetHost(SharedParameters, 3, 0.8  ); // r:     restitution coefficient   [-]
}
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "ImpactDynamics_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Solver Configuration
#define SOLVER RKCK45     // RK4, RKCK45
#define PRECISION double  // float, double
const int NT   = 30720; // NumberOfThreads
const int SD   = 3;     // SystemDimension
const int NCP  = 1;     // NumberOfControlParameters
const int NSP  = 4;     // NumberOfSharedParameters
const int NISP = 0;     // NumberOfIntegerSharedParameters
const int NE   = 2;     // NumberOfEvents
const int NA   = 3;     // NumberOfAccessories
const int NIA  = 2;     // NumberOfIntegerAccessories
const int NDO  = 0;     // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&);

int main()
{
	// The control parameter
	int NumberOfFlowRates = NT;
	
	vector<PRECISION> FlowRates(NumberOfFlowRates,0);
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
	
	
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanPressureReliefValve(SelectedDevice);
	
	ScanPressureReliefValve.SolverOption(ThreadsPerBlock, 64);
	ScanPressureReliefValve.SolverOption(RelativeTolerance, 0, 1.0e-10);
	ScanPressureReliefValve.SolverOption(RelativeTolerance, 1, 1.0e-10);
	ScanPressureReliefValve.SolverOption(RelativeTolerance, 2, 1.0e-10);
	ScanPressureReliefValve.SolverOption(AbsoluteTolerance, 0, 1.0e-10);
	ScanPressureReliefValve.SolverOption(AbsoluteTolerance, 1, 1.0e-10);
	ScanPressureReliefValve.SolverOption(AbsoluteTolerance, 2, 1.0e-10);
	ScanPressureReliefValve.SolverOption(EventDirection,   0, -1);
	ScanPressureReliefValve.SolverOption(EventDirection,   1, -1);
	
	
	// Simulation
	int NumberOfSimulationLaunches = NumberOfProblems / NumberOfThreads;
	int ProblemStartIndex;
	
	ofstream DataFile;
	DataFile.open ( "PressureReliefValve.txt" );
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);
	
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		// Fill Solver Object
		ProblemStartIndex = LaunchCounter * NumberOfThreads;
		FillSolverObject(ScanPressureReliefValve, FlowRates); // There is only 1 launch; thus, no special care is needed!
		ScanPressureReliefValve.SynchroniseFromHostToDevice(All);
		
		
		// Transient simulations
		TransientStart = clock();
		for (int i=0; i<1024; i++)
		{
			ScanPressureReliefValve.Solve();
			ScanPressureReliefValve.InsertSynchronisationPoint();
			ScanPressureReliefValve.SynchroniseSolver();
		}
		TransientEnd = clock();
			cout << "Launches: " << LaunchCounter << "  Simulation time: " << 1000.0*(TransientEnd-TransientStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
		
		// Converged simulations and their data collection
		for (int i=0; i<32; i++)
		{
			ScanPressureReliefValve.Solve();
			ScanPressureReliefValve.SynchroniseFromDeviceToHost(Accessories);
			ScanPressureReliefValve.InsertSynchronisationPoint();
			ScanPressureReliefValve.SynchroniseSolver();
			
			for (int tid=0; tid<NumberOfThreads; tid++)
			{
				DataFile.width(Width); DataFile << ScanPressureReliefValve.GetHost<PRECISION>(tid, ControlParameters, 0) << ',';
				DataFile.width(Width); DataFile << ScanPressureReliefValve.GetHost<PRECISION>(tid, Accessories, 0) << ',';
				DataFile.width(Width); DataFile << ScanPressureReliefValve.GetHost<PRECISION>(tid, Accessories, 1) << ',';
				DataFile << '\n';
			}
		}
	}
	
	DataFile.close();
	
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
}

// ------------------------------------------------------------------------------------------------

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
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

// ------------------------------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& q_Values)
{	
	int ProblemNumber = 0;
	for (auto const& q: q_Values) // dimensionless flow rate [-]
	{
		Solver.SetHost(ProblemNumber, TimeDomain, 0, 0.0);
		Solver.SetHost(ProblemNumber, TimeDomain, 1, 1.0e10); // Stopped by Poincaré section
		
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
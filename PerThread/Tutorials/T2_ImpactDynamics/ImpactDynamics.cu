/*
Second tutorial example: T2 (Impact Dynamics)
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "MassivelyParallel_GPU-ODE_Solver.cuh"

#define PI 3.14159265358979323846

using namespace std;

void Linspace(vector<double>&, double, double, int);
void Logspace(vector<double>&, double, double, int);

void FillProblemPool(ProblemPool&, const vector<double>&);

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
	int PoolSize        = NumberOfFlowRates; // 30720
	int NumberOfThreads = NumberOfFlowRates; // 30720 -> 1 launches
	
	ConstructorConfiguration ConfigurationPressureReliefValve;
	
	ConfigurationPressureReliefValve.PoolSize                  = PoolSize;
	ConfigurationPressureReliefValve.NumberOfThreads           = NumberOfThreads;
	ConfigurationPressureReliefValve.SystemDimension           = 3;
	ConfigurationPressureReliefValve.NumberOfControlParameters = 1;
	ConfigurationPressureReliefValve.NumberOfSharedParameters  = 4;
	ConfigurationPressureReliefValve.NumberOfEvents            = 2;
	ConfigurationPressureReliefValve.NumberOfAccessories       = 2;
	
	CheckStorageRequirements(ConfigurationPressureReliefValve, SelectedDevice);
	
	ProblemSolver ScanPressureReliefValve(ConfigurationPressureReliefValve, SelectedDevice);
	ProblemPool ProblemPoolPressureReliefValve(ConfigurationPressureReliefValve);
	
	FillProblemPool(ProblemPoolPressureReliefValve, FlowRates);
	
	//ProblemPoolPressureReliefValve.Print(TimeDomain);
	//ProblemPoolPressureReliefValve.Print(ActualState);
	//ProblemPoolPressureReliefValve.Print(ControlParameters);
	//ProblemPoolPressureReliefValve.Print(SharedParameters);
	//ProblemPoolPressureReliefValve.Print(Accessories);
	
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
	int NumberOfSimulationLaunches = PoolSize / NumberOfThreads;
	
	SolverConfiguration SolverConfigurationSystem;
		SolverConfigurationSystem.BlockSize       = 64;
		SolverConfigurationSystem.InitialTimeStep = 1e-2;
		SolverConfigurationSystem.Solver          = RKCK45;
		SolverConfigurationSystem.ActiveThreads   = NumberOfThreads;
	
	int CopyStartIndexInPool;
	int CopyStartIndexInSolverObject = 0;
	int NumberOfElementsCopied       = NumberOfThreads;
	
	ofstream DataFile;
	DataFile.open ( "PressureReliefValve.txt" );
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);
	
	
	clock_t SimulationStart = clock();
	ScanPressureReliefValve.SharedCopyFromPoolHostAndDevice(ProblemPoolPressureReliefValve);
	
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		CopyStartIndexInPool = LaunchCounter * NumberOfThreads;
		ScanPressureReliefValve.LinearCopyFromPoolHostAndDevice(ProblemPoolPressureReliefValve, CopyStartIndexInPool, CopyStartIndexInSolverObject, NumberOfElementsCopied, All);
		
		
		// Transient simulations
		for (int i=0; i<1024; i++)
			ScanPressureReliefValve.Solve(SolverConfigurationSystem);
		
		
		// Converged simulations and their data collection
		for (int i=0; i<32; i++)
		{
			ScanPressureReliefValve.Solve(SolverConfigurationSystem);
			
			for (int tid=0; tid<NumberOfThreads; tid++)
			{
				DataFile.width(Width); DataFile << ScanPressureReliefValve.SingleGetHost(tid, ControlParameters, 0) << ',';
				DataFile.width(Width); DataFile << ScanPressureReliefValve.SingleGetHost(tid, Accessories, 0) << ',';
				DataFile.width(Width); DataFile << ScanPressureReliefValve.SingleGetHost(tid, Accessories, 1) << ',';
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

void FillProblemPool(ProblemPool& Pool, const vector<double>& q_Values)
{	
	int ProblemNumber = 0;
	for (auto const& q: q_Values) // dimensionless flow rate [-]
	{
		Pool.Set(ProblemNumber, TimeDomain, 0, 0);
		Pool.Set(ProblemNumber, TimeDomain, 1, 1e10); // Stopped by Poincaré section
		
		Pool.Set(ProblemNumber, ActualState, 0, 0.2);
		Pool.Set(ProblemNumber, ActualState, 1, 0.0);
		Pool.Set(ProblemNumber, ActualState, 2, 0.0);
		
		Pool.Set(ProblemNumber, ControlParameters,  0, q);
		
		ProblemNumber++;
	}
	
	Pool.SetShared(0, 1.25 ); // Kappa: damping coefficient       [-]
	Pool.SetShared(1, 10.0 ); // Delta: spring precompression     [-]
	Pool.SetShared(2, 20.0 ); // Beta:  compressibility parameter [-]
	Pool.SetShared(3, 0.8  ); // r:     restitution coefficient   [-]
}
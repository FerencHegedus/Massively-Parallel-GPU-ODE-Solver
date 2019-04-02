/*
Bifurcation diagram of the Duffing oscillator: ddx + k*dx + x + x^3 = B*cos(t).
It is decomposed into 2D first order system: x1=x, x2=dx.
There are two parameters: k and B where B is kept fixed, while k is varied between 0.2 and 0.3 with resolution of 46080.
Therefore, tehere are altogether 46080 independent Duffing equation need to be solved. One GPU thread solve one Duffing system.
Since B is constant, that is, it is shared among all the systems, it is loaded also into the shared memory (in order to keep generality it is also stored as an ordinary parameter).
The Poincare sections (32 points) are stored after 1024 number of iterations. One iteration means integration of the system between 0 and 2*pi.
Event is detected according to the following event function: x2=0. With negative direction. It means that every local maxima of x1 is detected. It is also stored to a special storage variable called Accessories.
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
void FillProblemPool(ProblemPool&, const vector<double>&, double, double, double);


int main()
{
	int PoolSize        = 46081;
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
	
	CheckStorageRequirements(ConfigurationDuffing, SelectedDevice);
	
	ProblemSolver ScanDuffing(ConfigurationDuffing, SelectedDevice);
	
	ProblemPool ProblemPoolDuffing(ConfigurationDuffing);
		FillProblemPool(ProblemPoolDuffing, Parameters_k_Values, Parameters_B, InitialConditions_X1, InitialConditions_X2);
	
	//ProblemPoolDuffing.Print(TimeDomain);
	//ProblemPoolDuffing.Print(ActualState);
	//ProblemPoolDuffing.Print(ControlParameters);
	//ProblemPoolDuffing.Print(SharedParameters);
	//ProblemPoolDuffing.Print(Accessories);
	
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
	int NumberOfSimulationLaunches = PoolSize / NumberOfThreads + (PoolSize % NumberOfThreads == 0 ? 0:1);
	
	
	SolverConfiguration SolverConfigurationSystem;
		SolverConfigurationSystem.BlockSize       = BlockSize;
		SolverConfigurationSystem.InitialTimeStep = 1e-2;
		SolverConfigurationSystem.Solver          = RKCK45;
		SolverConfigurationSystem.ActiveThreads   = NumberOfThreads;
	
	
	int CopyStartIndexInPool;
	int CopyStartIndexInSolverObject = 0;
	
	ofstream DataFile;
	DataFile.open ( "Duffing.txt" );
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);
	
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	
	ScanDuffing.SharedCopyFromPoolHostAndDevice(ProblemPoolDuffing);
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		CopyStartIndexInPool = LaunchCounter * NumberOfThreads;
		
		if ( LaunchCounter == (NumberOfSimulationLaunches-1) )
			SolverConfigurationSystem.ActiveThreads = (PoolSize % NumberOfThreads == 0 ? NumberOfThreads : PoolSize % NumberOfThreads);
		
		
		ScanDuffing.LinearCopyFromPoolHostAndDevice(ProblemPoolDuffing, CopyStartIndexInPool, CopyStartIndexInSolverObject, SolverConfigurationSystem.ActiveThreads, All);
		
		TransientStart = clock();
		for (int i=0; i<1024; i++)
		{
			ScanDuffing.Solve(SolverConfigurationSystem);
		}
		TransientEnd = clock();
			cout << "Transient iteration: " << LaunchCounter << "  Simulation time: " << 1000.0*(TransientEnd-TransientStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
		
		for (int i=0; i<32; i++)
		{
			ScanDuffing.Solve(SolverConfigurationSystem);
			
			for (int tid=0; tid<SolverConfigurationSystem.ActiveThreads; tid++)
			{
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, ControlParameters, 0) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SharedGetHost(0) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, ActualState, 0) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, ActualState, 1) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, Accessories, 0) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, Accessories, 1) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, Accessories, 2) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(tid, Accessories, 3) << ',';
				DataFile << '\n';
			}
		}
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

void FillProblemPool(ProblemPool& Pool, const vector<double>& k_Values, double B, double X10, double X20)
{
	int ProblemNumber = 0;
	for (auto const& k: k_Values)
	{
		Pool.Set(ProblemNumber, TimeDomain,  0, 0 );
		Pool.Set(ProblemNumber, TimeDomain,  1, 2*PI );
		
		Pool.Set(ProblemNumber, ActualState, 0, X10 );
		Pool.Set(ProblemNumber, ActualState, 1, X20 );
		
		Pool.Set(ProblemNumber, ControlParameters, 0, k );
		
		Pool.Set(ProblemNumber, Accessories, 0, 0 );
		Pool.Set(ProblemNumber, Accessories, 1, 0 );
		Pool.Set(ProblemNumber, Accessories, 2, 0 );
		Pool.Set(ProblemNumber, Accessories, 3, 1e-2 );
		
		ProblemNumber++;
	}
	
	Pool.SetShared(0, B );
}
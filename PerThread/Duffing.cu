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

#include "ParametricODESolver.cuh"

#define gpuErrCHK(call)                                                                \
{                                                                                      \
	const cudaError_t error = call;                                                    \
	if (error != cudaSuccess)                                                          \
	{                                                                                  \
		cout << "Error: " << __FILE__ << ":" << __LINE__ << endl;                      \
		cout << "code:" << error << ", reason: " << cudaGetErrorString(error) << endl; \
		exit(1);                                                                       \
	}                                                                                  \
}

#define PI 3.14159265358979323846

using namespace std;

void Linspace(vector<double>&, double, double, int);
void FillProblemPool(ProblemPool&, const vector<double>&, double, double, double);



int main()
{
	int PoolSize        = 46080;
	int NumberOfThreads = PoolSize;
	int BlockSize       = 64;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
		gpuErrCHK( cudaSetDevice( SelectedDevice ) );
	
	PrintPropertiesOfTheSelectedDevice(SelectedDevice);
	
	
	
	double InitialConditions_X1 = -0.5;
	double InitialConditions_X2 = -0.1;
	double Parameters_B = 0.3;
	
	int NumberOfParameters_k = NumberOfThreads;
	double kRangeLower = 0.2;
    double kRangeUpper = 0.3;
		vector<double> Parameters_k_Values(NumberOfParameters_k,0);
		Linspace(Parameters_k_Values, kRangeLower, kRangeUpper, NumberOfParameters_k);
	
	
	
	ConstructorConfiguration ConfigurationDuffing;
		ConfigurationDuffing.PoolSize                  = PoolSize;
		ConfigurationDuffing.NumberOfThreads           = NumberOfThreads;
		ConfigurationDuffing.SystemDimension           = 2;
		ConfigurationDuffing.NumberOfControlParameters = 2;
		ConfigurationDuffing.NumberOfSharedParameters  = 1;
		ConfigurationDuffing.NumberOfEvents            = 1;
		ConfigurationDuffing.NumberOfAccessories       = 1;
	
	CheckStorageRequirements(ConfigurationDuffing, SelectedDevice);
	
	ProblemSolver ScanDuffing(ConfigurationDuffing);
	
	ProblemPool ProblemPoolDuffing(ConfigurationDuffing);
		FillProblemPool(ProblemPoolDuffing, Parameters_k_Values, Parameters_B, InitialConditions_X1, InitialConditions_X2);
	
	ProblemPoolDuffing.Print(TimeDomain);
	ProblemPoolDuffing.Print(ActualState);
	ProblemPoolDuffing.Print(ControlParameters);
	ProblemPoolDuffing.Print(SharedParameters);
	ProblemPoolDuffing.Print(Accessories);
	
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
	int NumberOfSimulationLaunches = PoolSize / NumberOfThreads; // Only one launch is required as PoolSize=NumberOfThreads.
	
	SolverConfiguration SolverConfigurationSystem;
		SolverConfigurationSystem.BlockSize       = BlockSize;
		SolverConfigurationSystem.InitialTimeStep = 1e-2;
		SolverConfigurationSystem.Solver          = RKCK45;
	
	int CopyStartIndexInPool;
	int CopyStartIndexInSolverObject = 0;
	int NumberOfElementsCopied       = NumberOfThreads;
	
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
			ScanDuffing.LinearCopyFromPoolHostAndDevice(ProblemPoolDuffing, CopyStartIndexInPool, CopyStartIndexInSolverObject, NumberOfElementsCopied, All);
		
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
			
			for (int idx=0; idx<NumberOfThreads; idx++)
			{
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(idx, ControlParameters, 0) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(idx, ControlParameters, 1) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(idx, ActualState, 0) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(idx, ActualState, 1) << ',';
				DataFile.width(Width); DataFile << ScanDuffing.SingleGetHost(idx, Accessories, 0) << ',';
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
		Pool.Set(ProblemNumber, ControlParameters, 1, B ); // Also stored in shared
		
		Pool.Set(ProblemNumber, Accessories, 0, 0 );
		
		ProblemNumber++;
	}
	
	Pool.SetShared(0, B );
}
/*
First tutaorial example: T1 (Quasiperiodic Forcing)
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "MassivelyParallel_GPU-ODE_Solver.cuh"

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
void Logspace(vector<double>&, double, double, int);

void FillProblemPool(ProblemPool&, const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&);

int main()
{
	// Physical control parameters
	int NumberOfFrequency1 = 128;
	int NumberOfFrequency2 = 128;
	int NumberOfAmplitude1 = 2;
	int NumberOfAmplitude2 = 2;
	
	vector<double> Frequency1(NumberOfFrequency1,0);
	vector<double> Frequency2(NumberOfFrequency2,0);
	vector<double> Amplitude1(NumberOfAmplitude1,0);
	vector<double> Amplitude2(NumberOfAmplitude2,0);
	
	Logspace(Frequency1, 20.0, 1000.0, NumberOfFrequency1);
	Logspace(Frequency2, 20.0, 1000.0, NumberOfFrequency2);
	Linspace(Amplitude1,  0.5,    1.1, NumberOfAmplitude1);
	Linspace(Amplitude2,  0.7,    1.2, NumberOfAmplitude2);
	
	
	// Setup CUDA a device
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
		gpuErrCHK( cudaSetDevice( SelectedDevice ) );
	
	PrintPropertiesOfTheSelectedDevice(SelectedDevice);
	
	
	// Problem Pool and Solver Object configuration
	int PoolSize        = NumberOfFrequency1 * NumberOfFrequency2 * NumberOfAmplitude1 * NumberOfAmplitude2; // 65536
	int NumberOfThreads = NumberOfFrequency1 * NumberOfFrequency2; // 16384 -> 4 launches
	
	ConstructorConfiguration ConfigurationKellerMiksis;
	
	ConfigurationKellerMiksis.PoolSize                  = PoolSize;
	ConfigurationKellerMiksis.NumberOfThreads           = NumberOfThreads;
	ConfigurationKellerMiksis.SystemDimension           = 2;
	ConfigurationKellerMiksis.NumberOfControlParameters = 21;
	ConfigurationKellerMiksis.NumberOfSharedParameters  = 0;
	ConfigurationKellerMiksis.NumberOfEvents            = 1;
	ConfigurationKellerMiksis.NumberOfAccessories       = 4;
	
	CheckStorageRequirements(ConfigurationKellerMiksis, SelectedDevice);
	
	ProblemSolver ScanKellerMiksis(ConfigurationKellerMiksis);
	ProblemPool ProblemPoolKellerMiksis(ConfigurationKellerMiksis);
	
	FillProblemPool(ProblemPoolKellerMiksis, Frequency1, Frequency2, Amplitude1, Amplitude2);
	
	ProblemPoolKellerMiksis.Print(TimeDomain);
	ProblemPoolKellerMiksis.Print(ActualState);
	ProblemPoolKellerMiksis.Print(ControlParameters);
	ProblemPoolKellerMiksis.Print(SharedParameters);
	ProblemPoolKellerMiksis.Print(Accessories);
	
	
// SIMULATIONS ------------------------------------------------------------------------------------
	
	int NumberOfSimulationLaunches = PoolSize / NumberOfThreads;
	
	SolverConfiguration SolverConfigurationSystem;
		SolverConfigurationSystem.BlockSize       = 64;
		SolverConfigurationSystem.InitialTimeStep = 1e-2;
		SolverConfigurationSystem.Solver          = RKCK45;
	
	int CopyStartIndexInPool;
	int CopyStartIndexInSolverObject = 0;
	int NumberOfElementsCopied       = NumberOfThreads;
	
	vector< vector<double> > CollectedData;
	CollectedData.resize( NumberOfThreads , vector<double>( 136 , 0 ) );
		// 136 =  6 physical parameters +
		//        1 initial time of the converged iterations +
		//        1 total time of the converged iterations +
		//       64 local maxima +
		//       64 local minima.
	
	double ActualPA1;
	double ActualPA2;
	clock_t SimulationStart = clock();
	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		// Copy data from Problem Pool to Solver Object
		CopyStartIndexInPool = LaunchCounter * NumberOfThreads;
		ScanKellerMiksis.LinearCopyFromPoolHostAndDevice(ProblemPoolKellerMiksis, CopyStartIndexInPool, CopyStartIndexInSolverObject, NumberOfElementsCopied, All);
		
		
		// Generate a unique filename for the current launch
		stringstream StreamFilename;
		StreamFilename.precision(2);
		StreamFilename.setf(ios::fixed);
		
		ActualPA1 = ScanKellerMiksis.SingleGetHost(0, ControlParameters, 15);
		ActualPA2 = ScanKellerMiksis.SingleGetHost(0, ControlParameters, 17);
		StreamFilename << "KellerMiksis_Collapse_PA1_" << ActualPA1 << "_PA2_" << ActualPA2 << ".txt";
		
		string Filename = StreamFilename.str();
		remove( Filename.c_str() );
		
		
		// Collect physical parameters
		for (int tid=0; tid<NumberOfThreads; tid++)
		{
			CollectedData[tid][0] = ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 15);
			CollectedData[tid][1] = ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 16);
			CollectedData[tid][2] = ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 17);
			CollectedData[tid][3] = ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 18);
			CollectedData[tid][4] = ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 19);
			CollectedData[tid][5] = ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 20);
		}
		
		
		// Transient simulations
		for (int i=0; i<1024; i++)
		//{
		//	cout << "Launch: " << LaunchCounter << "  Transient: " << i << endl;
			ScanKellerMiksis.Solve(SolverConfigurationSystem);
		//}
		
		
		// Collect the initial time of the converged iteration
		for (int tid=0; tid<NumberOfThreads; tid++)
			CollectedData[tid][6] = ScanKellerMiksis.SingleGetHost(tid, TimeDomain, 0) * ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 13); // Convert to [s]
		
		
		// Converged simulations and their data collection
		for (int i=0; i<64; i++)
		{
		//	cout << "Launch: " << LaunchCounter << "  Converged: " << i << endl;
			ScanKellerMiksis.Solve(SolverConfigurationSystem);
			
			for (int tid=0; tid<NumberOfThreads; tid++)
			{	
				CollectedData[tid][8+i]    = ScanKellerMiksis.SingleGetHost(tid, Accessories, 0); // Local maxima
				CollectedData[tid][8+i+64] = ScanKellerMiksis.SingleGetHost(tid, Accessories, 2); // Local minima
			}
		}
		
		
		// Collect the total time of the converged iterations
		for (int tid=0; tid<NumberOfThreads; tid++)
			CollectedData[tid][7] = ScanKellerMiksis.SingleGetHost(tid, TimeDomain, 0) * ScanKellerMiksis.SingleGetHost(tid, ControlParameters, 13) - CollectedData[tid][6];
		
		
		// Save collected data to file
		ofstream DataFile;
		DataFile.open ( Filename.c_str(), std::fstream::app );
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);
		
		for (int tid=0; tid<NumberOfThreads; tid++)
		{
			for (int col=0; col<136; col++)
			{
				if ( col<(136-1) )
				{
					DataFile.width(Width); DataFile << CollectedData[tid][col] << ',';
				} else
				{
					DataFile.width(Width); DataFile << CollectedData[tid][col];
				}
			}
			DataFile << '\n';
		}
		
		DataFile.close();
		
	}
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

void FillProblemPool(ProblemPool& Pool, const vector<double>& F1_Values, const vector<double>& F2_Values, const vector<double>& PA1_Values, const vector<double>& PA2_Values)
{
	// Declaration of physical control parameters
	double P1; // pressure amplitude1 [bar]
	double P2; // frequency1          [kHz]
	double P3; // pressure amplitude2 [bar]
	double P4; // frequency2          [kHz]
	
	// Declaration of constant parameters
	double P5 =  0.0; // phase shift          [-]
	double P6 = 10.0; // equilibrium radius   [mum]
	double P7 =  1.0; // ambient pressure     [bar]
	double P9 =  1.4; // polytrophic exponent [-]
	
	// Material properties
	double Pv  = 3.166775638952003e+03;
    double Rho = 9.970639504998557e+02;
    double ST  = 0.071977583160056;
    double Vis = 8.902125058209557e-04;
    double CL  = 1.497251785455527e+03;
	
	// Auxiliary variables
	double Pinf;
	double PA1;
	double PA2;
	double RE;
	double f1;
	double f2;
	
	int ProblemNumber = 0;
	for (auto const& CP4: PA2_Values) // pressure amplitude2 [bar]
	{
		for (auto const& CP3: PA1_Values) // pressure amplitude1 [bar]
		{
			for (auto const& CP2: F2_Values) // frequency2 [kHz]
			{
				for (auto const& CP1: F1_Values) // frequency1 [kHz]
				{
					// Update physical parameters
					P1 = CP3;
					P2 = CP1;
					P3 = CP4;
					P4 = CP2;
					
					Pool.Set(ProblemNumber, TimeDomain, 0, 0);
					Pool.Set(ProblemNumber, TimeDomain, 1, 1e10);
					
					// Initial conditions are the equilibrium condition y1=1; y2=0;
					Pool.Set(ProblemNumber, ActualState, 0, 1.0);
					Pool.Set(ProblemNumber, ActualState, 1, 0.0);
					
					// Scaling of physical parameters to SI
					Pinf = P7 * 1e5;
					PA1  = P1 * 1e5;
					PA2  = P3 * 1e5;
					RE   = P6 / 1e6;
					
					// Scale to angular frequency
					f1   = 2.0*PI*(P2*1000);
					f2   = 2.0*PI*(P4*1000);
					
					// System coefficients and other, auxiliary parameters
					Pool.Set(ProblemNumber, ControlParameters,  0, (2.0*ST/RE + Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Pool.Set(ProblemNumber, ControlParameters,  1, (1.0-3.0*P9) * (2*ST/RE + Pinf - Pv) * (2.0*PI/RE/f1) / CL/Rho );
					Pool.Set(ProblemNumber, ControlParameters,  2, (Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Pool.Set(ProblemNumber, ControlParameters,  3, (2.0*ST/RE/Rho) * pow(2.0*PI/RE/f1, 2.0) );
					Pool.Set(ProblemNumber, ControlParameters,  4, (4.0*Vis/Rho/pow(RE,2.0)) * (2.0*PI/f1) );
					Pool.Set(ProblemNumber, ControlParameters,  5, PA1 * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Pool.Set(ProblemNumber, ControlParameters,  6, PA2 * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Pool.Set(ProblemNumber, ControlParameters,  7, (RE*f1*PA1/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) );
					Pool.Set(ProblemNumber, ControlParameters,  8, (RE*f2*PA2/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) );
					Pool.Set(ProblemNumber, ControlParameters,  9, RE*f1/(2.0*PI)/CL );
					Pool.Set(ProblemNumber, ControlParameters, 10, 3.0*P9 );
					Pool.Set(ProblemNumber, ControlParameters, 11, P4/P2 );
					Pool.Set(ProblemNumber, ControlParameters, 12, P5 );
					
					Pool.Set(ProblemNumber, ControlParameters, 13, 2.0*PI/f1 ); // tref
					Pool.Set(ProblemNumber, ControlParameters, 14, RE );        // Rref
					
					Pool.Set(ProblemNumber, ControlParameters, 15, P1 );
					Pool.Set(ProblemNumber, ControlParameters, 16, P2 );
					Pool.Set(ProblemNumber, ControlParameters, 17, P3 );
					Pool.Set(ProblemNumber, ControlParameters, 18, P4 );
					Pool.Set(ProblemNumber, ControlParameters, 19, P5 );
					Pool.Set(ProblemNumber, ControlParameters, 20, P6 );
					
					ProblemNumber++;
				}
			}
		}
	}
}
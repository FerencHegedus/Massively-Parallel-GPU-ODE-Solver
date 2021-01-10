#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

// Solver Configuration
#define __MPGOS_PERTHREAD_ALGORITHM 1     // RK4 - 0, RKCK45 - 1
#define __MPGOS_PERTHREAD_NT 46080  	// NumberOfThreads
#define __MPGOS_PERTHREAD_SD 2				// SystemDimension
#define __MPGOS_PERTHREAD_NCP 1				// NumberOfControlParameters
#define __MPGOS_PERTHREAD_NSP 1				// NumberOfSharedParameters

#include "SingleSystem_PerThread_DataStructures.cuh"
#include "Poincare_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

void Linspace(vector<double>&, double, double, int);
void FillSolverObject(ProblemSolver&, const vector<double>&, double, double, double, int, int);
void SaveData(ProblemSolver&, ofstream&, int);

int main()
{
	int NumberOfProblems = 1<<12;
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


	ProblemSolver ScanDuffing(SelectedDevice);

	ScanDuffing.SolverOption(ThreadsPerBlock, BlockSize);
	ScanDuffing.SolverOption(InitialTimeStep, 1.0e-2);
	ScanDuffing.SolverOption(ActiveNumberOfThreads, __MPGOS_PERTHREAD_NT);

	ScanDuffing.SolverOption(MaximumTimeStep, 1.0e3);
	ScanDuffing.SolverOption(MinimumTimeStep, 1.0e-14);
	ScanDuffing.SolverOption(TimeStepGrowLimit, 10.0);
	ScanDuffing.SolverOption(TimeStepShrinkLimit, 0.2);

	ScanDuffing.SolverOption(RelativeTolerance, 0, 1.0e-9);
	ScanDuffing.SolverOption(RelativeTolerance, 1, 1.0e-9);
	ScanDuffing.SolverOption(AbsoluteTolerance, 0, 1.0e-9);
	ScanDuffing.SolverOption(AbsoluteTolerance, 1, 1.0e-9);


	int NumberOfSimulationLaunches = NumberOfProblems / __MPGOS_PERTHREAD_NT + (NumberOfProblems % __MPGOS_PERTHREAD_NT == 0 ? 0:1);

	ofstream DataFile;
	DataFile.open ( "DuffingPoincare.txt" );

	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;

	for (int LaunchCounter=0; LaunchCounter<NumberOfSimulationLaunches; LaunchCounter++)
	{
		FillSolverObject(ScanDuffing, Parameters_k_Values, Parameters_B, InitialConditions_X1, InitialConditions_X2, LaunchCounter * __MPGOS_PERTHREAD_NT, __MPGOS_PERTHREAD_NT);

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

			SaveData(ScanDuffing, DataFile, __MPGOS_PERTHREAD_NT);
		}
	}

	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;

	DataFile.close();

	cout << "Test finished!" << endl;
}

// AUXILIARY FUNCTION -----------------------------------------------------------------------------

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

void FillSolverObject(ProblemSolver& Solver, const vector<double>& k_Values, double B, double X10, double X20, int FirstProblemNumber, int NumberOfThreads)
{
	int k_begin = FirstProblemNumber;
	int k_end   = FirstProblemNumber + NumberOfThreads;

	int ProblemNumber = 0;
	for (int k=k_begin; k<k_end; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0.0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, 2.0*PI );

		Solver.SetHost(ProblemNumber, ActualState, 0, X10 );
		Solver.SetHost(ProblemNumber, ActualState, 1, X20 );

		Solver.SetHost(ProblemNumber, ControlParameters, 0, k_Values[k] );

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
		DataFile.width(Width); DataFile << Solver.GetHost<double>(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<double>(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<double>(tid, ActualState, 1);
		DataFile << '\n';
	}
}

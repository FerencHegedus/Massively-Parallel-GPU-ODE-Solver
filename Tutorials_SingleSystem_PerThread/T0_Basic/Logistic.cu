#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

// Solver Configuration
#define __MPGOS_PERTHREAD_ALGORITHM 0 //RK4 solver
#define __MPGOS_PERTHREAD_PRECISION float
#define __MPGOS_PERTHREAD_NT    46080 // NumberOfThreads
#define __MPGOS_PERTHREAD_SD    1     // SystemDimension

#include "SingleSystem_PerThread_DataStructures.cuh"
#include "Logistic_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

using namespace std;

void Linspace(vector<float>&, float, float, int);

int main()
{
	//run configuration
  int NumberOfProblems = __MPGOS_PERTHREAD_NT;
  int BlockSize = 64;
  int CUDAdevice = 0;
  PrintPropertiesOfSpecificDevice(CUDAdevice);

	//parameters and initial conditions
	float tmin = 		0.0;
	float tmax = 		10.0;
	float dt = 			0.05;
  float x0min = 	0;
  float x0max = 	10;
  vector<float> x0(NumberOfProblems,0);
	Linspace(x0,x0min,x0max, NumberOfProblems);

	//initialize solver
	ProblemSolver Solver(CUDAdevice);

	//Set Solver options
	Solver.SolverOption(ThreadsPerBlock, BlockSize);
	Solver.SolverOption(InitialTimeStep, dt);
	Solver.SolverOption(ActiveNumberOfThreads, NumberOfProblems);

	//fill solver object
	for (int i = 0; i < NumberOfProblems; i++)
	{
		Solver.SetHost(i,TimeDomain,0,tmin);
		Solver.SetHost(i,TimeDomain,1,tmax);
		Solver.SetHost(i,ActualTime,tmin);

		Solver.SetHost(i,ActualState,0,x0[i]);
	}

	//synchronize
	Solver.SynchroniseFromHostToDevice(All);
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();

	//solve
	clock_t SimulationStart = clock();
	Solver.Solve();
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();
	clock_t SimulationTime = clock()-SimulationStart;
	std::cout << "Simulation Time: "<< 1000*SimulationTime/CLOCKS_PER_SEC << " ms"<<std::endl;

	//write back to CPU
	Solver.SynchroniseFromDeviceToHost(All);
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();

	//write to file
	ofstream DataFile("logistic.txt");
	DataFile.precision(8);
	DataFile.flags(ios::scientific);
	for (int i = 0; i < NumberOfProblems; i++)
	{
		DataFile.width(13); DataFile << x0[i] << ',';
		DataFile.width(13); DataFile << Solver.GetHost<float>(i, ActualState, 0) << '\n';
	}
	DataFile.flush();
	DataFile.close();

	cout << "Test finished!" << endl;
}

void Linspace(vector<float>& x, float B, float E, int N)
{
    float Increment;

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

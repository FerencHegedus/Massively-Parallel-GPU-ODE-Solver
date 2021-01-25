/*
Simulation of the simpified SIR-Modell
https://link.springer.com/article/10.1007/s12064-019-00300-7
with A=0, mu=0, sigma=0, omega=0, phi=0

This tutorial calculates the maximal number of infected people depending on different
protective measures beta. Where beta=[0,200]. The time interval is t=[0,180]

initial values are s=0.9999, i=0.0001, r=0.0 constants, thus their derivatives are 0
only 2 initial points are necessary
*/
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

// Solver Configuration
#define __MPGOS_PERTHREAD_SOLVER_DDE4 //DDE4 solver
#define __MPGOS_PERTHREAD_PRECISION double
#define __MPGOS_PERTHREAD_NT    		46080 // NumberOfThreads
#define __MPGOS_PERTHREAD_SD    		3     // SystemDimension
#define __MPGOS_PERTHREAD_DOD    		1     // DenseDimension
#define __MPGOS_PERTHREAD_NDELAY    1     // NumberOfDelays
#define __MPGOS_PERTHREAD_NCP    		1			// ControlParameters
#define __MPGOS_PERTHREAD_NSP       2     // Shared parameters
#define __MPGOS_PERTHREAD_NA        2     // Accessories
#define __MPGOS_PERTHREAD_NDO   		100   // NumberOfPointsOfDenseOutput

#include "SingleSystem_PerThread_DataStructures.cuh"
#include "Epidemic_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

using namespace std;

void Linspace(vector<double>&, double, double);

int main()
{
  //run configuration
  int NumberOfProblems = __MPGOS_PERTHREAD_NT;
  int BlockSize = 32;
  int CUDAdevice = 0;
  PrintPropertiesOfSpecificDevice(CUDAdevice);

  //time domain
	double tmin = 0.0;
	double tmax = 180.0;
	double dt = 0.1;

	//initial conditions
	double s0 = 0.9999;	//suspectible at t=0
	double i0 = 0.0001; //infectious  at t=0
	double r0 = 0.0;		//recovered   at t=0

	//parameters
  double alpha = 1.1; //transmission rate
  double theta = 0.7; //recovery rate
  double tau = 7.0;   //incubation time
	vector <double> beta(NumberOfProblems); //protective measures
	Linspace(beta,0,200.0);


	//initialize solver
	ProblemSolver Solver(CUDAdevice);
	Solver.SolverOption(ThreadsPerBlock, BlockSize);
	Solver.SolverOption(InitialTimeStep, dt);
	Solver.SolverOption(ActiveNumberOfThreads, NumberOfProblems);

	Solver.SolverOption(DenseOutputTimeStep, -1); //save every point in the dense output
	Solver.SolverOption(DenseOutputVariableIndex, 0, 1); //0. dense output -> 1. system variable

	Solver.SolverOption(Delay, 0, 0, tau); //0. delay -> 0. dense output (-> 1. system variable)

	//fill solver object
	Solver.SetHost(SharedParameters, 0, alpha);
	Solver.SetHost(SharedParameters, 1, theta);


	for (int i = 0; i < NumberOfProblems; i++)
	{
		Solver.SetHost(i,TimeDomain,0,tmin);
		Solver.SetHost(i,TimeDomain,1,tmax);
		Solver.SetHost(i,ActualTime,tmin);

		Solver.SetHost(i,ActualState,0,s0);
		Solver.SetHost(i,ActualState,1,i0);
		Solver.SetHost(i,ActualState,2,r0);
		Solver.SetHost(i,ControlParameters,0,beta[i]);

		//fill initial dense output
		Solver.SetHost(i,DenseIndex,2);
		Solver.SetHost(i,DenseTime,0,-tau);
		Solver.SetHost(i,DenseTime,1,tmin);
		Solver.SetHost(i,DenseState,0,0,i0);
		Solver.SetHost(i,DenseState,0,1,i0);
		Solver.SetHost(i,DenseDerivative,0,0,0.0);
		Solver.SetHost(i,DenseDerivative,0,1,0.0);
	}

	//synchronize
	Solver.SynchroniseFromHostToDevice(All);
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();

	//solve
	clock_t SimulationStart = clock();

	Solver.Solve();
	Solver.SynchroniseFromDeviceToHost(All);
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();

	clock_t SimulationTime = clock()-SimulationStart;
	std::cout << "Simulation Time: "<< 1000*SimulationTime/CLOCKS_PER_SEC << " ms"<<std::endl;

	//save data
	ofstream DataFile;
	DataFile.open ("epidemic.txt");
	DataFile.precision(14);
	DataFile.flags(ios::scientific);

	for (size_t tid = 0; tid < __MPGOS_PERTHREAD_NT; tid++)
	{
		DataFile.width(20); DataFile << Solver.GetHost<double>(tid,ControlParameters,0) << ",";
		DataFile.width(20); DataFile << Solver.GetHost<double>(tid,Accessories,0) << ",";
		DataFile.width(20); DataFile << Solver.GetHost<double>(tid,Accessories,1) << "\n";
	}


	DataFile.flush();
	DataFile.close();

	Solver.Print(DenseOutput,0,1);

  return 0;
}

void Linspace(vector<double>& x, double B, double E)
{
  double Increment;
	int N = x.size();
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

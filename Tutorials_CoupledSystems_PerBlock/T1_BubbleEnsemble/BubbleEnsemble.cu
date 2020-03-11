#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <random>

#include "CoupledSystems_PerBlock.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Physical control parameters
const int NumberOfFrequency      = 101; // Control parameter
const int NumberOfAmplitude      = 101; // Control parameter
const int NumberOfUnitsPerSystem = 90;  // Number coupled units

// Solver Configuration
#define SOLVER RK4 // RK4, RKCK45
const int NS   = NumberOfFrequency * NumberOfAmplitude; // NumberOfSystems
const int UPS  = NumberOfUnitsPerSystem;                // UnitsPerSystem
const int UD   = 2;     // UnitDimension
const int TPB  = 32;    // ThreadsPerBlock
const int SPB  = 5;     // SystemPerBlock
const int NC   = 1;     // NumberOfCouplings

const int NUP  = 21;    // NumberOfUnitParameters (different form system to system, different from unit to unit)
const int NSP  = 0;     // NumberOfSystemParameters (different from system to system, shared by all units)
const int NGP  = 0;     // NumberOfGlobalParameters (shared by all systems, share by all units)
const int NiGP = 0;     // NumberOfIntegerGlobalParameters (shared by all systems, shared by all units)

const int NUA  = 1;     // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 0;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 0;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 0;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 0;     // NumberOfEvents (per units)
const int NDO  = 100;   // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<double>&, double, double, int);
void Logspace(vector<double>&, double, double, int);
void Random(vector<double>&, double, double, int, int);
void Gauss(vector<double>&, double, double, int);

//void FillSolverObject(ProblemSolver&, const vector<double>&, const vector<double>&, const vector<double>&);
//void FillCouplingMatrix(ProblemSolver&, const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&);

int main()
{
	vector<double> Frequency(NumberOfFrequency,0);
	vector<double> Amplitude(NumberOfAmplitude,0);
	vector<double> BubbleSize(NumberOfUnitsPerSystem,0);
	vector<double> PositionX(NumberOfUnitsPerSystem,0);
	vector<double> PositionY(NumberOfUnitsPerSystem,0);
	vector<double> PositionZ(NumberOfUnitsPerSystem,0);
	
	Logspace(Frequency, 20.0, 1000.0, NumberOfFrequency);  // kHz
	Linspace(Amplitude,  0.0,    0.8, NumberOfAmplitude);  // bar
	Random(BubbleSize,   1.0,    5.0, NumberOfUnitsPerSystem, 10001); // micron
	Gauss(PositionX,     0.0,   10.0, NumberOfUnitsPerSystem); // mm
	Gauss(PositionY,     0.0,   10.0, NumberOfUnitsPerSystem); // mm
	Gauss(PositionZ,     0.0,   10.0, NumberOfUnitsPerSystem); // mm
	
	BubbleSize[0]=10.0;
	BubbleSize[1]=8.0;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,float> ScanSystem(SelectedDevice);
	
	//ScanSystem.SolverOption(InitialTimeStep,1.52);
	//ScanSystem.SolverOption(EventStopCounter,5,1e-6);
	
	
	//FillSolverObject(ScanSystem, Frequency, Amplitude, BubbleSize);
	//FillCouplingMatrix(ScanSystem, PositionX, PositionY, PositionZ, BubbleSize);
	
	// Print Positions
	/*for (int Col=0; Col<NumberOfUnitsPerSystem; Col++)
	{
		std::cout.width(6);
		cout << std::setprecision(3) << PositionX[Col] << " ";
		std::cout.width(6);
		cout << std::setprecision(3) << PositionY[Col] << " ";
		std::cout.width(6);
		cout << std::setprecision(3) << PositionZ[Col] << " " << endl;
	}
	cout << endl;*/
	
	
	// Print CouplingMatrix
	/*for (int Row=0; Row<NumberOfUnitsPerSystem; Row++)
	{
		for (int Col=0; Col<NumberOfUnitsPerSystem; Col++)
		{
			std::cout.width(8);
			cout << std::setprecision(3) << ScanSystem.GetHost(CouplingMatrix, Row, Col) << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	
	
	/*ScanSystem.Print(TimeDomain);
	ScanSystem.Print(ActualState);
	ScanSystem.Print(ControlParameters);
	ScanSystem.Print(SharedParameters);
	ScanSystem.Print(CouplingMatrix);
	
	
	ScanSystem.SynchroniseFromHostToDevice(All);
	
	ScanSystem.SolverOption(RK4_EH0_SSSBL, 1e-5, ConfigurationDuffing.NumberOfSystems);
	
	// Initial data
	int SystemNumber = 1000;
	int UnitNumber   = 0;
	
	cout << "Bubble radius1: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ControlParameters, 20) << endl;
	cout << "Bubble radius2: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber+1, ControlParameters, 20) << endl;
	cout << "Amplitude:     " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ControlParameters, 15) << endl;
	cout << "Frequency:     " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ControlParameters, 16) << endl << endl;
	
	cout << "Initial state X1: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ActualState, 0) << endl;
	cout << "Initial state X2: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ActualState, 1) << endl << endl;
	cout << "Initial state X3: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber+1, ActualState, 0) << endl;
	cout << "Initial state X4: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber+1, ActualState, 1) << endl << endl;
	
	ScanSystem.Solve();
	ScanSystem.SynchroniseFromDeviceToHost(All);
	ScanSystem.InsertSynchronisationPoint();
	ScanSystem.SynchroniseSolver();
	
	cout << "Final state X1: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ActualState, 0) << endl;
	cout << "Final state X2: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber, ActualState, 1) << endl << endl;
	cout << "Final state X3: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber+1, ActualState, 0) << endl;
	cout << "Final state X4: " << std::setprecision(15) << ScanSystem.GetHost(SystemNumber, UnitNumber+1, ActualState, 1) << endl << endl;
	*/
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

void Random(vector<double>& x, double B, double E, int N, int Res)
{
    srand(time(NULL));
	
	int Random;
	
	for (int i=0; i<N; i++)
	{
		Random = (rand() % Res);
		x[i] = B + Random*(E-B)/Res;
	}
}

void Gauss(vector<double>& x, double M, double D, int N)
{
    default_random_engine generator;
	normal_distribution<double> distribution(M,D);
	
	for (int i=0; i<N; i++)
	{
		x[i] = distribution(generator);
	}
}

// ------------------------------------------------------------------------------------------------

/*void FillSolverObject(ProblemSolver& Solver, const vector<double>& Frequency, const vector<double>& Amplitude, const vector<double>& BubbleSize)
{
	double P1;   // pressure amplitude1 [bar] (Shared among units, different among blocks; the first control parameter)
	double P2;   // relative frequency1 [kHz] (Shared among units, different among blocks; the second control parameter)
	double P3=0; // pressure amplitude2 [bar] (Zero)
	double P4=0; // relative frequency2 [kHz] (Zero)
	double P5=0; // phase shift         [-]   (Zero)
	double P6;   // equilibrium radius  [mum] (Shared among blocks, different among units; NOT a control parameter, describes the bubble size distribution)
	
	double P7 = 1.0; // ambient pressure     [bar]
	double P9 = 1.4; // polytrophic exponent [-]
	
	double Pv  = 3.166775638952003e+03;
    double Rho = 9.970639504998557e+02;
    double ST  = 0.071977583160056;
    double Vis = 8.902125058209557e-04;
    double CL  = 1.497251785455527e+03;
	
	double Pinf;
	double PA1;
	double PA2;
	double RE;
	double f1;
	double f2;
	
	
	// Loop over the control parameters (blocks)
	int SystemNumber = 0;
	for (auto const& CP2: Frequency) // Frequency [kHz]
	{
		for (auto const& CP1: Amplitude) // Amplitude [bar]
		{
			// Loop over the equilibrium bubble radii (units)
			// Unit scope
			int UnitNumber   = 0;
			for (auto const& CP0: BubbleSize) // equilibrium radius  [mum]
			{
				// Update parameters
				P1 = CP1;
				P2 = CP2;
				P6 = CP0;
				
				// Fill up time domains
				//Solver.SetHost(SystemNumber, UnitNumber, TimeDomain, 0, 0);
				//Solver.SetHost(SystemNumber, UnitNumber, TimeDomain, 1, 1);
				
				// Fill up initial conditions
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, 1.0);
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 1, 0.0);
				
				// Fill up parameters
				Pinf = P7 * 1e5;
				PA1  = P1 * 1e5;
				PA2  = P3 * 1e5;
				RE   = P6 / 1e6;
				
				f1   = 2.0*PI*(P2*1000);
				f2   = 2.0*PI*(P4*1000);
				
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  0, (2.0*ST/RE + Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  1, (1.0-3.0*P9) * (2*ST/RE + Pinf - Pv) * (2.0*PI/RE/f1) / CL/Rho );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  2, (Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  3, (2.0*ST/RE/Rho) * pow(2.0*PI/RE/f1, 2.0) );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  4, (4.0*Vis/Rho/pow(RE,2.0)) * (2.0*PI/f1) );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  5, PA1 * pow(2.0*PI/RE/f1, 2.0) / Rho );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  6, PA2 * pow(2.0*PI/RE/f1, 2.0) / Rho );          // ZERO: Shared
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  7, (RE*f1*PA1/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  8, (RE*f2*PA2/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) ); // ZERO: Shared
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters,  9, RE*f1/(2.0*PI)/CL );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 10, 3.0*P9 );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 11, P4/P2 ); // ZERO: Shared
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 12, P5 );    // ZERO: Shared
				
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 13, 2.0*PI/f1 ); // tref
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 14, RE );        // Rref
				
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 15, P1 );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 16, P2 );
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 17, P3 ); // ZERO
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 18, P4 ); // ZERO
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 19, P5 ); // ZERO
				Solver.SetHost(SystemNumber, UnitNumber, ControlParameters, 20, P6 );
				
				UnitNumber++;
			}
			
			// System scope
			Solver.SetHost(SystemNumber, TimeDomain, 0, 0 );
			Solver.SetHost(SystemNumber, TimeDomain, 1, 1.02 );
			
			SystemNumber++;
		}
	}
	
	// Global scope
	Solver.SetHost(SharedParameters, 0, 0.0 ); // CP6
	Solver.SetHost(SharedParameters, 1, 0.0 ); // CP8
	Solver.SetHost(SharedParameters, 2, 0.0 ); // CP11
	Solver.SetHost(SharedParameters, 3, 0.0 ); // CP12
	
	// Dummy global scope
	for (int i=4; i<(4+32+32); i++)
	{
		Solver.SetHost(SharedParameters, i, i ); // Dummy Shared
	}
}

void FillCouplingMatrix(ProblemSolver& Solver, const vector<double>& X, const vector<double>& Y, const vector<double>& Z, const vector<double>& RE)
{
	int N = X.size();
	double Distance;
	
	for (int Row=0; Row<N; Row++)
	{
		for (int Col=0; Col<N; Col++)
		{
			if ( Row != Col )
			{
				//Distance = sqrt( (X[Row]-X[Col])*(X[Row]-X[Col]) + (Y[Row]-Y[Col])*(Y[Row]-Y[Col]) + (Z[Row]-Z[Col])*(Z[Row]-Z[Col]) ) * 1000;
				Distance = 50;
				Solver.SetHost(CouplingMatrix, Row, Col, pow(RE[Col],3) / pow(RE[Row],2) / Distance);
			} else
			{
				Solver.SetHost(CouplingMatrix, Row, Col, 0);
			}
		}
	}
}*/
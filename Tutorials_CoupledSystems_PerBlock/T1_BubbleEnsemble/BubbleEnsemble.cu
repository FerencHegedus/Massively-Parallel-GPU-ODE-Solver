#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <random>

#include "BubbleEnsemble_SystemDefinition.cuh"
#include "CoupledSystems_PerBlock_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Physical control parameters
const int NumberOfFrequency      = 1; // Control parameter
const int NumberOfAmplitude      = 8; // Control parameter
const int NumberOfUnitsPerSystem = 12;  // Number coupled units

// Solver Configuration
#define SOLVER RK4      // RK4, RKCK45
#define PRECISION double // float, double
const int NS   = NumberOfFrequency * NumberOfAmplitude; // NumberOfSystems
const int UPS  = NumberOfUnitsPerSystem;                // UnitsPerSystem
const int UD   = 2;     // UnitDimension
const int TPB  = 32;    // ThreadsPerBlock (integer multiple of the warp size that is 32)
const int SPB  = 3;     // SystemPerBlock
const int NC   = 2;     // NumberOfCouplings

const int NUP  = 21;    // NumberOfUnitParameters (different form system to system, different from unit to unit)
const int NSP  = 1;     // NumberOfSystemParameters (different from system to system, shared by all units)
const int NGP  = 20;     // NumberOfGlobalParameters (shared by all systems, share by all units)
const int NiGP = 35;     // NumberOfIntegerGlobalParameters (shared by all systems, shared by all units)

const int NUA  = 1;     // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 2;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 3;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 4;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 3;     // NumberOfEvents (per units)
const int NDO  = 100;   // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Random(vector<PRECISION>&, PRECISION, PRECISION, int, int);
void Gauss(vector<PRECISION>&, PRECISION, PRECISION, int);

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);

int main()
{
	vector<PRECISION> Frequency(NumberOfFrequency,0);
	vector<PRECISION> Amplitude(NumberOfAmplitude,0);
	vector<PRECISION> BubbleSize(NumberOfUnitsPerSystem,0);
	vector<PRECISION> PositionX(NumberOfUnitsPerSystem,0);
	vector<PRECISION> PositionY(NumberOfUnitsPerSystem,0);
	vector<PRECISION> PositionZ(NumberOfUnitsPerSystem,0);
	
	Logspace(Frequency, 20.0, 1000.0, NumberOfFrequency);  // kHz
	Linspace(Amplitude,  0.0,    0.8, NumberOfAmplitude);  // bar
	Random(BubbleSize,   1.0,    5.0, NumberOfUnitsPerSystem, 10001); // micron
	Gauss(PositionX,     0.0,   10.0, NumberOfUnitsPerSystem); // mm
	Gauss(PositionY,     0.0,   10.0, NumberOfUnitsPerSystem); // mm
	Gauss(PositionZ,     0.0,   10.0, NumberOfUnitsPerSystem); // mm
	
	//BubbleSize[0]=10.0;
	//BubbleSize[1]=8.0;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION> ScanSystem(SelectedDevice);
	
	ScanSystem.SolverOption(SharedCouplingMatrices, 1);
	ScanSystem.SolverOption(SharedGlobalVariables,  1);
	//ScanSystem.SolverOption(SharedGlobalVariables,  1);
	
	ScanSystem.SolverOption(RelativeTolerance, 0, 1e-10);
	
	FillSolverObject(ScanSystem, Frequency, Amplitude, BubbleSize);
	FillCouplingMatrix(ScanSystem, PositionX, PositionY, PositionZ, BubbleSize);
	
	//ScanSystem.Print(DenseOutput,0);
	
	
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
			cout << std::setprecision(3) << ScanSystem.GetHost<PRECISION>(0, CouplingMatrix, Row, Col) << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	
	//ScanSystem.Print(TimeDomain);
	//ScanSystem.Print(ActualState);
	ScanSystem.Print(UnitParameters);
	ScanSystem.Print(CouplingMatrix,0);
	ScanSystem.Print(CouplingMatrix,1);
	//ScanSystem.Print(IntegerSystemAccessories);
	//ScanSystem.Print(CouplingStrength);
	//ScanSystem.Print(CouplingIndex);
	
	ScanSystem.SynchroniseFromHostToDevice(All);
	
	
	
	ScanSystem.Solve();
	
	//cout << ScanSystem.GetHost<int>(IntegerGlobalParameters, 0) << endl;
	//cout << ScanSystem.GetHost<int>(IntegerGlobalParameters, 1) << endl;
	//cout << ScanSystem.GetHost<int>(IntegerGlobalParameters, 2) << endl;
	
	// Initial data
	/*int SystemNumber = 1000;
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

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
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

void Random(vector<PRECISION>& x, PRECISION B, PRECISION E, int N, int Res)
{
    srand(time(NULL));
	
	int Random;
	
	for (int i=0; i<N; i++)
	{
		Random = (rand() % Res);
		x[i] = B + Random*(E-B)/Res;
	}
}

void Gauss(vector<PRECISION>& x, PRECISION M, PRECISION D, int N)
{
    default_random_engine generator;
	normal_distribution<PRECISION> distribution(M,D);
	
	for (int i=0; i<N; i++)
	{
		x[i] = distribution(generator);
	}
}

// ------------------------------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& Frequency, const vector<PRECISION>& Amplitude, const vector<PRECISION>& BubbleSize)
{
	PRECISION P1;   // pressure amplitude1 [bar] (Shared among units, different among blocks; the first control parameter)
	PRECISION P2;   // relative frequency1 [kHz] (Shared among units, different among blocks; the second control parameter)
	PRECISION P3=0; // pressure amplitude2 [bar] (Zero)
	PRECISION P4=0; // relative frequency2 [kHz] (Zero)
	PRECISION P5=0; // phase shift         [-]   (Zero)
	PRECISION P6;   // equilibrium radius  [mum] (Shared among blocks, different among units; NOT a control parameter, describes the bubble size distribution)
	
	PRECISION P7 = 1.0; // ambient pressure     [bar]
	PRECISION P9 = 1.4; // polytrophic exponent [-]
	
	PRECISION Pv  = 3.166775638952003e+03;
    PRECISION Rho = 9.970639504998557e+02;
    PRECISION ST  = 0.071977583160056;
    PRECISION Vis = 8.902125058209557e-04;
    PRECISION CL  = 1.497251785455527e+03;
	
	PRECISION Pinf;
	PRECISION PA1;
	PRECISION PA2;
	PRECISION RE;
	PRECISION f1;
	PRECISION f2;
	
	
	// Loop over the systems (blocks): control parameters
	int SystemNumber = 0;
	for (auto const& CP2: Frequency) // Frequency [kHz]
	{
		for (auto const& CP1: Amplitude) // Amplitude [bar]
		{
			// SYSTEM SCOPE
			Solver.SetHost(SystemNumber, TimeDomain, 0, 0.0);
			Solver.SetHost(SystemNumber, TimeDomain, 1, 1.0);
			
			for (int i=0; i<NC; i++)
				Solver.SetHost(SystemNumber, CouplingStrength, i, 1.0);
			
			// DUMMY System Parameters ----------------------------------------
			for (int i=0; i<NSP; i++)
				Solver.SetHost(SystemNumber, SystemParameters, i, (SystemNumber+1)*10+(i+1));
			
			for (int i=0; i<NSA; i++)
				Solver.SetHost(SystemNumber, SystemAccessories, i, (SystemNumber+1)*10+(i+1));
			
			for (int i=0; i<NiSA; i++)
				Solver.SetHost(SystemNumber, IntegerSystemAccessories, i, (SystemNumber+1)*10+(i+1));
			
			for (int i=0; i<NDO; i++)
				Solver.SetHost(SystemNumber, DenseTime, i, i);
			
				Solver.SetHost(SystemNumber, DenseIndex, SystemNumber+1);
			// ----------------------------------------------------------------
			
			
			// UNIT SCOPE
			int UnitNumber = 0;
			for (auto const& CP0: BubbleSize) // equilibrium radius  [mum]
			{
				P1 = CP1; // pressure amplitude1 [bar]
				P2 = CP2; // relative frequency1 [kHz]
				P6 = CP0; // equilibrium radius  [mum]
				
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, 1.0);
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 1, 0.0);
				
				// Dimensional physical parameters
				Pinf = P7 * 1e5;
				PA1  = P1 * 1e5;
				PA2  = P3 * 1e5;
				RE   = P6 / 1e6;
				
				f1   = 2.0*PI*(P2*1000);
				f2   = 2.0*PI*(P4*1000);
				
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  0, (2.0*ST/RE + Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  1, (1.0-3.0*P9) * (2*ST/RE + Pinf - Pv) * (2.0*PI/RE/f1) / CL/Rho );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  2, (Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  3, (2.0*ST/RE/Rho) * pow(2.0*PI/RE/f1, 2.0) );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  4, (4.0*Vis/Rho/pow(RE,2.0)) * (2.0*PI/f1) );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  5, PA1 * pow(2.0*PI/RE/f1, 2.0) / Rho );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  6, PA2 * pow(2.0*PI/RE/f1, 2.0) / Rho );          // ZERO: Shared
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  7, (RE*f1*PA1/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  8, (RE*f2*PA2/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) ); // ZERO: Shared
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  9, RE*f1/(2.0*PI)/CL );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 10, 3.0*P9 );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 11, P4/P2 ); // ZERO: Shared
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 12, P5 );    // ZERO: Shared
				
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 13, 2.0*PI/f1 ); // tref
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 14, RE );        // Rref
				
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 15, P1 );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 16, P2 );
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 17, P3 ); // ZERO
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 18, P4 ); // ZERO
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 19, P5 ); // ZERO
				Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 20, P6 );
				
				// DUMMY Unit Parameters --------------------------------------
				for (int i=0; i<NUA; i++)
					Solver.SetHost(SystemNumber, UnitNumber, UnitAccessories, i, (SystemNumber+1)*100 + (UnitNumber+1)*10 + (i+1));
				
				for (int i=0; i<NiUA; i++)
					Solver.SetHost(SystemNumber, UnitNumber, IntegerUnitAccessories, i, (SystemNumber+1)*100 + (UnitNumber+1)*10 + (i+1));
				
				for (int i=0; i<NDO; i++)
				{
					Solver.SetHost(SystemNumber, UnitNumber, DenseState, 0, i, i);
					Solver.SetHost(SystemNumber, UnitNumber, DenseState, 1, i, i);
				}
				// ------------------------------------------------------------
				
				UnitNumber++;
			}
			
			SystemNumber++;
		}
	}
	
	// GLOBAL SCOPE
	for (int i=0; i<NC; i++)
		Solver.SetHost(CouplingIndex, i, 1);
	
	// DUMMY Global Parameters ------------------------------------
	for (int i=0; i<NGP; i++)
		Solver.SetHost(GlobalParameters, i, 5.0+i);
	
	for (int i=0; i<NiGP; i++)
		Solver.SetHost(IntegerGlobalParameters, i, 10.0+i);
	// ------------------------------------------------------------
}

void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& X, const vector<PRECISION>& Y, const vector<PRECISION>& Z, const vector<PRECISION>& RE)
{
	int N = X.size();
	PRECISION Distance;
	
	for (int Row=0; Row<N; Row++)
	{
		for (int Col=0; Col<N; Col++)
		{
			if ( Row != Col )
			{
				//Distance = 50;
				Distance = sqrt( (X[Row]-X[Col])*(X[Row]-X[Col]) + (Y[Row]-Y[Col])*(Y[Row]-Y[Col]) + (Z[Row]-Z[Col])*(Z[Row]-Z[Col]) ) * 1000; // Change units from [mm] to [m]
				Solver.SetHost(0, CouplingMatrix, Row, Col, pow(RE[Col],3) / pow(RE[Row],2) / Distance);
			} else
			{
				Solver.SetHost(0, CouplingMatrix, Row, Col, 0);
			}
		}
	}
}
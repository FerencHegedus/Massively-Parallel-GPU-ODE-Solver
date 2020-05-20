#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <random>

#include "BubbleEnsemble_SystemDefinition.cuh"
#include "CoupledSystems_PerBlock_Interface.cuh"

#define PI 3.14159265358979323846

#define COUPLINGSTRENGTH 1.0
#define DISTANCE 2000.0;

using namespace std;

// Physical control parameters
const int NumberOfFrequency      = 1; // Control parameter
const int NumberOfAmplitude      = 120*5; // Control parameter
const int NumberOfUnitsPerSystem = 128; // Number coupled units

// Solver Configuration
#define SOLVER RKCK45      // RK4, RKCK45
#define PRECISION double   // float, double
const int NS   = NumberOfFrequency * NumberOfAmplitude; // NumberOfSystems
const int UPS  = NumberOfUnitsPerSystem;                // UnitsPerSystem
const int UD   = 2;     // UnitDimension
const int TPB  = 128;    // ThreadsPerBlock (integer multiple of the warp size that is 32)
const int SPB  = 1;     // SystemsPerBlock
const int NC   = 1;     // NumberOfCouplings
const int CBW  = 0;     // CouplingBandwidthRadius (0: full coupling matrix)
const int CCI  = 0;     // CouplingCircularity (0: non-circular matrix, 1: circular matrix)

const int NUP  = 21;    // NumberOfUnitParameters (different form system to system, different from unit to unit)
const int NSP  = 0;     // NumberOfSystemParameters (different from system to system, shared by all units)
const int NGP  = 0;     // NumberOfGlobalParameters (shared by all systems, share by all units)
const int NiGP = 0;     // NumberOfIntegerGlobalParameters (shared by all systems, shared by all units)

const int NUA  = 0;     // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 0;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 0;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 0;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 0;     // NumberOfEvents (per units)
const int NDO  = 0;   // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Random(vector<PRECISION>&, PRECISION, PRECISION, int, int);
void Gauss(vector<PRECISION>&, PRECISION, PRECISION, int);

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);

void SavePositions(vector<PRECISION>&, vector<PRECISION>&, vector<PRECISION>&, string, int);
void SaveBubbleSizes(vector<PRECISION>&, string, int);
void SaveData(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, ofstream&, int, int, int);

// ------------------------------------------------------------------------------------------------

int main()
{
	vector<PRECISION> Frequency(NumberOfFrequency,0);
	vector<PRECISION> Amplitude(NumberOfAmplitude,0);
	vector<PRECISION> BubbleSize(UPS,0);
	vector<PRECISION> PositionX(UPS,0);
	vector<PRECISION> PositionY(UPS,0);
	vector<PRECISION> PositionZ(UPS,0);
	
	Logspace(Frequency, 1000.0, 2000.0, NumberOfFrequency);  // kHz
	Linspace(Amplitude,   0.0,    5.0, NumberOfAmplitude);  // bar
	Random(BubbleSize,    4.0,    4.0, UPS, 10001); // micron
	Gauss(PositionX,      0.0,   10.0, UPS); // mm
	Gauss(PositionY,      0.0,   10.0, UPS); // mm
	Gauss(PositionZ,      0.0,   10.0, UPS); // mm
	
	//BubbleSize[0]=10.0;
	//BubbleSize[1]=8.0;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION> ScanSystem(SelectedDevice);
	
	ScanSystem.SolverOption(SharedCouplingMatrices, 0);
	ScanSystem.SolverOption(InitialTimeStep, 1e-3);
	ScanSystem.SolverOption(RelativeTolerance, 0, 1e-10);
	ScanSystem.SolverOption(AbsoluteTolerance, 0, 1e-10);
	ScanSystem.SolverOption(MinimumTimeStep, 1e-13);
	
	FillSolverObject(ScanSystem, Frequency, Amplitude, BubbleSize);
	FillCouplingMatrix(ScanSystem, PositionX, PositionY, PositionZ, BubbleSize);
	
	SavePositions(PositionX, PositionY, PositionZ, "Positions.txt", UPS);
	SaveBubbleSizes(BubbleSize, "BubbleSizes.txt", UPS);
	
	ScanSystem.Print(TimeDomain);
	ScanSystem.Print(ActualState);
	ScanSystem.Print(UnitParameters);
	ScanSystem.Print(CouplingMatrix,0);
	//ScanSystem.Print(CouplingMatrix,1);
	
	
	// CHECK COUPLING MATRIX
	/*cout << ScanSystem.GetHost<PRECISION>(0, CouplingMatrix, 0, 0) << " " << endl;
	cout << endl;
	cout << ScanSystem.GetHost<PRECISION>(1, CouplingMatrix, 0, 0) << " " << endl;*/
	
	
	
	
	
	
	// SIMULATION -------------------------------------------------------------
	
	ofstream DataFile;
	
	string       Filename;
	stringstream StreamFilename;
	
	PRECISION Tmp1 = DISTANCE;
	PRECISION Tmp2 = COUPLINGSTRENGTH;
	StreamFilename.str("");
	StreamFilename.precision(0);
	StreamFilename.setf(ios::fixed);
	//StreamFilename << "BubbleEnsemble_" << Frequency[0] << "_" << Tmp1 << "_" << Tmp2 << ".txt";
	StreamFilename << "TestCase_N64_1.txt";
	Filename = StreamFilename.str();
		
	DataFile.open ( Filename.c_str() );
	
	clock_t SimulationStart = clock();
	clock_t TransientStart;
	clock_t TransientEnd;
	
	ScanSystem.SynchroniseFromHostToDevice(All);
	// Transients
	for (int i=0; i<100; i++)
	{
		ScanSystem.Solve();
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		//cout << "Transient finished: " << i << endl;
	}
	
		ScanSystem.SynchroniseFromDeviceToHost(All);
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
		SaveData(ScanSystem, DataFile, NS, UPS, UD);
		
	// Save Poincare section
	/*for (int i=0; i<0; i++)
	{
		ScanSystem.Solve();
		ScanSystem.SynchroniseFromDeviceToHost(All);
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		SaveData(ScanSystem, DataFile, NS, UPS, UD);
		
		cout << "Poincare finished: " << i << endl;
	}*/
	
	
	
	DataFile.close();
	
	cout << "SIMULATION COMPLETED!" << endl;
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
    random_device rd;
	default_random_engine generator;
	generator.seed( rd() );
	normal_distribution<PRECISION> distribution(M,D);
	
	for (int i=0; i<N; i++)
	{
		distribution.reset();
		x[i] = distribution(generator);
	}
}

// ------------------------------------------------------------------------------------------------

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& Frequency, const vector<PRECISION>& Amplitude, const vector<PRECISION>& BubbleSize)
{
	srand(time(NULL));
	PRECISION LowerLimit = 0.75;
	PRECISION UpperLimit = 2.25;
	int Random;
	int Resolution = 100001;
	
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
				Solver.SetHost(SystemNumber, CouplingStrength, i, COUPLINGSTRENGTH);
			
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
				
				Random = (rand() % Resolution);
				//Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, LowerLimit + Random*(UpperLimit-LowerLimit)/Resolution );
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

void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& X, const vector<PRECISION>& Y, const vector<PRECISION>& Z, const vector<PRECISION>& RE)
{
	int N = X.size();
	PRECISION Distance;
	
	int ModRow;
	int ModCol;
	int ColLimit;
	
	for (int Row=0; Row<N; Row++)
	{
		for (int Col=0; Col<N; Col++)
		{
			// Full Matrix Storage---------------------------------------------
			if ( ( CBW == 0 ) && ( CCI == 0 ) )
			{
				if ( Row != Col )
				{
					Distance = DISTANCE;
					//Distance = sqrt( (X[Row]-X[Col])*(X[Row]-X[Col]) + (Y[Row]-Y[Col])*(Y[Row]-Y[Col]) + (Z[Row]-Z[Col])*(Z[Row]-Z[Col]) ) * 1000; // Change units from [mm] to [mum]
					Solver.SetHost(0, CouplingMatrix, Row, Col, pow(RE[Col],3) / pow(RE[Row],2) / Distance);
					
					if ( Distance < 100 )
						cout << Distance << endl;
					
					// DUMMY second coupling matrix
					//Solver.SetHost(1, CouplingMatrix, Row, Col, 1.125);
				} else
				{
					Solver.SetHost(0, CouplingMatrix, Row, Col, 0.0);
					
					// DUMMY second coupling matrix
					//Solver.SetHost(1, CouplingMatrix, Row, Col, 2.5);
				}
			}
			
			// Diagonal Matrix Storage-----------------------------------------
			if ( ( CBW > 0 ) && ( CCI == 0 ) )
			{
				ModRow = Row;
				ModCol = Col - Row + CBW;
				
				if ( ModCol >= UPS )
					ModCol = ModCol - UPS;
				
				if ( ModCol < 0 )
					ModCol = ModCol + UPS;
				
				if ( ( Row != Col ) && ( ModCol < 2*CBW+1 ) )
				{
					Distance = DISTANCE;
					//Distance = sqrt( (X[Row]-X[Col])*(X[Row]-X[Col]) + (Y[Row]-Y[Col])*(Y[Row]-Y[Col]) + (Z[Row]-Z[Col])*(Z[Row]-Z[Col]) ) * 1000; // Change units from [mm] to [mum]
					Solver.SetHost(0, CouplingMatrix, Row, Col, pow(RE[Col],3) / pow(RE[Row],2) / Distance);
					if ( Distance < 100 )
						cout << Distance << endl;
					
					// DUMMY second coupling matrix
					//Solver.SetHost(1, CouplingMatrix, Row, Col, 1.125);
				}
				
				if ( Row == Col )
				{
					Solver.SetHost(0, CouplingMatrix, Row, Col, 0.0);
					
					// DUMMY second coupling matrix
					//Solver.SetHost(1, CouplingMatrix, Row, Col, 2.5);
				}
			}
			
			// Collapsed Diagonal Matrix Storage-------------------------------
			if ( CCI == 1 )
			{
				ModRow = Row;
				ModCol = Col - Row + CBW;
				
				if ( ModCol >= UPS )
					ModCol = ModCol - UPS;
				
				if ( ModCol < 0 )
					ModCol = ModCol + UPS;
				
				if ( CBW == 0 )
					ColLimit = UPS;
				else
					ColLimit = 2*CBW+1;
				
				if ( ( Row != Col ) && ( ModCol < ColLimit ) )
				{
					Distance = DISTANCE;
					//Distance = sqrt( (X[Row]-X[Col])*(X[Row]-X[Col]) + (Y[Row]-Y[Col])*(Y[Row]-Y[Col]) + (Z[Row]-Z[Col])*(Z[Row]-Z[Col]) ) * 1000; // Change units from [mm] to [mum]
					Solver.SetHost(0, CouplingMatrix, Row, Col, pow(RE[Col],3) / pow(RE[Row],2) / Distance);
					if ( Distance < 100 )
						cout << Distance << endl;
					
					// DUMMY second coupling matrix
					//Solver.SetHost(1, CouplingMatrix, Row, Col, 1.125);
				}
				
				if ( Row == Col )
				{
					Solver.SetHost(0, CouplingMatrix, Row, Col, 0.0);
					
					// DUMMY second coupling matrix
					//Solver.SetHost(1, CouplingMatrix, Row, Col, 2.5);
				}
			}
			
			// Original
			/*if ( Row != Col )
			{
				Distance = DISTANCE;
				//Distance = sqrt( (X[Row]-X[Col])*(X[Row]-X[Col]) + (Y[Row]-Y[Col])*(Y[Row]-Y[Col]) + (Z[Row]-Z[Col])*(Z[Row]-Z[Col]) ) * 1000; // Change units from [mm] to [mum]
				Solver.SetHost(0, CouplingMatrix, Row, Col, pow(RE[Col],3) / pow(RE[Row],2) / Distance);
				if ( Distance < 100 )
					cout << Distance << endl;
			} else
			{
				Solver.SetHost(0, CouplingMatrix, Row, Col, 0.0);
			}*/
			
			//cout << endl;
		}
	}
}


// ------------------------------------------------------------------------------------------------

void SavePositions(vector<PRECISION>& PositionX, vector<PRECISION>& PositionY, vector<PRECISION>& PositionZ, string FileName, int UPS)
{
	ofstream DataFile;
	DataFile.open ( FileName.c_str() );
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	DataFile.width(7); DataFile << "ser.No." << ',';
	DataFile.width(Width); DataFile << "pos. X" << ',';
	DataFile.width(Width); DataFile << "pos. Y" << ',';
	DataFile.width(Width); DataFile << "pos. Z" << ',';
	DataFile << '\n';
	
	for (int Col=0; Col<UPS; Col++)
	{
		DataFile.width(7); DataFile << Col << ',';
		DataFile.width(Width); DataFile << PositionX[Col] << ',';
		DataFile.width(Width); DataFile << PositionY[Col] << ',';
		DataFile.width(Width); DataFile << PositionZ[Col] << ',';
		DataFile << '\n';
	}
	cout << endl;
	
	DataFile.close();
}

void SaveBubbleSizes(vector<PRECISION>& BubbleSize, string FileName, int UPS)
{
	ofstream DataFile;
	DataFile.open ( FileName.c_str() );
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	DataFile.width(7); DataFile << "ser.No." << ',';
	DataFile.width(Width); DataFile << "size (mum)" << ',';
	DataFile << '\n';
	
	for (int Col=0; Col<UPS; Col++)
	{
		DataFile.width(7); DataFile << Col << ',';
		DataFile.width(Width); DataFile << BubbleSize[Col] << ',';
		DataFile << '\n';
	}
	cout << endl;
	
	DataFile.close();
}

void SaveData(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, ofstream& DataFile, int NS, int UPS, int UD)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	/*for (int sid=0; sid<NS; sid++)
	{
		DataFile.width(4); DataFile << sid << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, 0, UnitParameters, 15) << ',';
		for (int uid=0; uid<UPS; uid++)
		{
			for (int cmp=0; cmp<UD; cmp++)
			{
				DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, uid, ActualState, cmp) << ',';
			}
		}
		DataFile << '\n';
	}*/
	
	// Speciality for 2 bubbles
	for (int sid=0; sid<NS; sid++)
	{
		DataFile.width(4); DataFile << sid << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, 0, UnitParameters, 15) << ',';
		for (int uid=0; uid<UPS; uid++)
		{
			for (int cmp=0; cmp<1; cmp++)
			{
				DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, uid, ActualState, cmp) << ',';
			}
		}
		DataFile << '\n';
	}
}
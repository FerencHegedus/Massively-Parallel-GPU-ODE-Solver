#ifndef PARAMETRIC_ODE_SOLVER_H
#define PARAMETRIC_ODE_SOLVER_H

using namespace std;

template <class DataType>
DataType* AllocateHostMemory(int);

template <class DataType>
DataType* AllocateDeviceMemory(int);

enum VariableSelection{	All, TimeDomain, ActualState, ControlParameters, SharedParameters, Accessories };
enum SolverAlgorithms{ RKCK45, RK4, RK4_EH0, RKCK45_EH0};

// DEVICE SETTINGS ------------------------------

void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfTheSelectedDevice(int);

// STRUCTURES -----------------------------------

struct ConstructorConfiguration
{
	int PoolSize;
	int NumberOfThreads;
	
	int SystemDimension;
	int NumberOfControlParameters;
	int NumberOfSharedParameters;
	int NumberOfEvents;
	int NumberOfAccessories;
};

struct SolverConfiguration
{
	int BlockSize;
	double InitialTimeStep;
	SolverAlgorithms Solver;
};

struct IntegratorInternalVariables
{
	int SystemDimension;
	int NumberOfThreads;
	int NumberOfControlParameters;
	int NumberOfSharedParameters;
	int NumberOfEvents;
	int NumberOfAccessories;
	
	double* d_TimeDomain;
	double* d_ActualState;
	double* d_ControlParameters;
	double* d_SharedParameters;
	double* d_Accessories;
	
	double* d_State;
	double* d_Stages;
	
	double* d_NextState;
	
	double* d_Error;
	double* d_ActualTolerance;
	
	double* d_ActualEventValue;
	double* d_NextEventValue;
	int*    d_EventCounter;
	int*    d_EventEquilibriumCounter;
	
	double InitialTimeStep;
};

// REQUIREMENTS ---------------------------------

void CheckStorageRequirements(const ConstructorConfiguration&, int);

// CLASSES --------------------------------------

class ProblemSolver;

class ProblemPool
{
    friend class ProblemSolver;
	
	private:
        int PoolSize;
		
		int SystemDimension;
		int NumberOfControlParameters;
		int NumberOfSharedParameters;
		int NumberOfAccessories;
		
		double* p_TimeDomain;
		double* p_ActualState;
		double* p_ControlParameters;
		double* p_SharedParameters;
		double* p_Accessories;
		
	public:
        ProblemPool(const ConstructorConfiguration&);
		~ProblemPool();
		
		void Set(int, VariableSelection, int, double);
		void SetShared(int, double);
		
		double Get(int, VariableSelection, int);
		double GetShared(int);
		
		void Print(VariableSelection);
};

class ProblemSolver
{
    private:
		double  h_BT_RK4[1];
		double  h_BT_RKCK45[26];
		
		double* h_TimeDomain;
		double* h_ActualState;
		double* h_ControlParameters;
		double* h_SharedParameters;
		double* h_Accessories;
		
		IntegratorInternalVariables KernelParameters;
		
	public:
		ProblemSolver(const ConstructorConfiguration&);
		~ProblemSolver();
		
		void LinearCopyFromPoolHostAndDevice(const ProblemPool&, int, int, int, VariableSelection);
		void SharedCopyFromPoolHostAndDevice(const ProblemPool&);
		
		void SingleSetHost(int, VariableSelection, int, double);
		void SingleSetHostAndDevice(int, VariableSelection, int, double);
		
		void SetSharedHost(int, double);
		void SetSharedHostAndDevice(int, double);
		
		void SynchroniseFromHostToDevice(VariableSelection);
		void SynchroniseFromDeviceToHost(VariableSelection);
		
		void SynchroniseSharedFromHostToDevice();
		void SynchroniseSharedFromDeviceToHost();
		
		double SingleGetHost(int, VariableSelection, int);
		double SharedGetHost(int);
		
		void Print(VariableSelection);
		
		void Solve(const SolverConfiguration&);
};

#endif
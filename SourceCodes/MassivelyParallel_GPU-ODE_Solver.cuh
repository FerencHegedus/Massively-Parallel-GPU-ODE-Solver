#ifndef MASSIVELYPARALLEL_GPUODE_SOLVER_H
#define MASSIVELYPARALLEL_GPUODE_SOLVER_H

template <class DataType>
DataType* AllocateHostMemory(int);

template <class DataType>
DataType* AllocateHostPinnedMemory(int);

template <class DataType>
DataType* AllocateDeviceMemory(int);

enum VariableSelection{	All, TimeDomain, ActualState, ControlParameters, SharedParameters, Accessories, DenseOutput, DenseTime, DenseState };
enum ListOfSolverAlgorithms{ RKCK45, RK4, RK4_EH0, RKCK45_EH0};
enum ListOfSolverOptions{ ThreadsPerBlock, InitialTimeStep, Solver, ActiveNumberOfThreads, \
                          MaximumTimeStep, MinimumTimeStep, TimeStepGrowLimit, TimeStepShrinkLimit, MaxStepInsideEvent, MaximumNumberOfTimeSteps, \
						  RelativeTolerance, AbsoluteTolerance, \
						  EventTolerance, EventDirection, EventStopCounter, \
						  DenseOutputTimeStep, DenseOutputEnabled };


void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);


struct ConstructorConfiguration
{
	int NumberOfThreads;
	int SystemDimension;
	int NumberOfControlParameters;
	int NumberOfSharedParameters;
	int NumberOfEvents;
	int NumberOfAccessories;
	
	int DenseOutputNumberOfPoints;
};

struct IntegratorInternalVariables
{
	int NumberOfThreads;
	int SystemDimension;
	int NumberOfControlParameters;
	int NumberOfSharedParameters;
	int NumberOfEvents;
	int NumberOfAccessories;
	
	double* d_TimeDomain;
	double* d_ActualState;
	double* d_ControlParameters;
	double* d_SharedParameters;
	double* d_Accessories;
	
	double* d_RelativeTolerance;
	double* d_AbsoluteTolerance;
	double  MaximumTimeStep;
	double  MinimumTimeStep;
	double  TimeStepGrowLimit;
	double  TimeStepShrinkLimit;
	double* d_EventTolerance;
	int*    d_EventDirection;
	int*    d_EventStopCounter;
	int     MaxStepInsideEvent;
	
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
	int ActiveThreads;
	
	int    DenseOutputEnabled;
	int    DenseOutputNumberOfPoints;
	double DenseOutputTimeStep;
	
	int*    d_DenseOutputIndex;
	double* d_DenseOutputTimeInstances;
	double* d_DenseOutputStates;
	
	int    MaximumNumberOfTimeSteps;
};


class ProblemSolver
{
    private:
		int Device;
		cudaStream_t Stream;
		cudaEvent_t Event;
		
		size_t GlobalMemoryRequired;
		size_t GlobalMemoryFree;
		size_t GlobalMemoryTotal;
		
		int SizeOfTimeDomain;
		int SizeOfActualState;
		int SizeOfControlParameters;
		int SizeOfSharedParameters;
		int SizeOfAccessories;
		int SizeOfEvents;
		
		int SizeOfDenseOutputIndex;
		int SizeOfDenseOutputTimeInstances;
		int SizeOfDenseOutputStates;
		
		size_t DynamicSharedMemoryRKCK45;
		size_t DynamicSharedMemoryRKCK45_EH0;
		size_t DynamicSharedMemoryRK4;
		size_t DynamicSharedMemoryRK4_EH0;
		
		double  h_BT_RK4[1];
		double  h_BT_RKCK45[26];
		
		double* h_TimeDomain;
		double* h_ActualState;
		double* h_ControlParameters;
		double* h_SharedParameters;
		double* h_Accessories;
		
		int*    h_DenseOutputIndex;
		double* h_DenseOutputTimeInstances;
		double* h_DenseOutputStates;
		
		int GridSize;
		int BlockSize;
		
		ListOfSolverAlgorithms SolverType;
		
		IntegratorInternalVariables KernelParameters;
		
	public:
		ProblemSolver(const ConstructorConfiguration&, int);
		~ProblemSolver();
		
		void SetHost(int, VariableSelection, int, double);      // Problem scope
		void SetHost(int, VariableSelection, int, int, double); // Dense state
		void SetHost(VariableSelection, int, double);           // Global scope
		void SynchroniseFromHostToDevice(VariableSelection);
		void SynchroniseFromDeviceToHost(VariableSelection);
		double GetHost(int, VariableSelection, int);            // Problem scope
		double GetHost(int, VariableSelection, int, int);       // Dense state
		double GetHost(VariableSelection, int);                 // Global scope
		
		void Print(VariableSelection);
		void Print(VariableSelection, int);
		
		void SolverOption(ListOfSolverOptions, int);    // int type
		void SolverOption(ListOfSolverOptions, double); // double type
		void SolverOption(ListOfSolverOptions, int, int);    // Array of int
		void SolverOption(ListOfSolverOptions, int, double); // Array of double
		void SolverOption(ListOfSolverOptions, ListOfSolverAlgorithms); // ListOfSolverAlgorithms type
		void Solve();
		
		void SynchroniseDevice();
		void InsertSynchronisationPoint();
		void SynchroniseSolver();
};

#endif
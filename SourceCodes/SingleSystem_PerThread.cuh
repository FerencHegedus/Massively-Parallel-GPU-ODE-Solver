#ifndef SINGLESYSTEM_PERTHREAD_H
#define SINGLESYSTEM_PERTHREAD_H

#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#define gpuErrCHK(call)                                                                          \
{                                                                                                \
	const cudaError_t error = call;                                                              \
	if (error != cudaSuccess)                                                                    \
	{                                                                                            \
		std::cout << "Error: " << __FILE__ << ":" << __LINE__ << std::endl;                      \
		std::cout << "code:" << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
		exit(1);                                                                                 \
	}                                                                                            \
}

template <class DataType>
DataType* AllocateHostMemory(int);

template <class DataType>
DataType* AllocateHostPinnedMemory(int);

template <class DataType>
DataType* AllocateDeviceMemory(int);

enum Algorithms{ RK4=2, RKCK45=6 };

enum VariableSelection{	All, TimeDomain, ActualState, ControlParameters, SharedParameters, Accessories, DenseOutput, DenseTime, DenseState };
enum IntegerVariableSelection{	IntegerSharedParameters, IntegerAccessories };
enum ListOfSolverOptions{ ThreadsPerBlock, InitialTimeStep, ActiveNumberOfThreads, \
                          MaximumTimeStep, MinimumTimeStep, TimeStepGrowLimit, TimeStepShrinkLimit, MaxStepInsideEvent, MaximumNumberOfTimeSteps, \
						  RelativeTolerance, AbsoluteTolerance, \
						  EventTolerance, EventDirection, EventStopCounter, \
						  DenseOutputTimeStep };


void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);


struct IntegratorInternalVariables
{
	int NumberOfThreads;
	int SystemDimension;
	int NumberOfControlParameters;
	int NumberOfSharedParameters;
	int NumberOfEvents;
	int NumberOfAccessories;
	int NumberOfIntegerSharedParameters;
	int NumberOfIntegerAccessories;
	
	double* d_TimeDomain;
	double* d_ActualState;
	double* d_ControlParameters;
	double* d_SharedParameters;
	double* d_Accessories;
	int*    d_IntegerSharedParameters;
	int*    d_IntegerAccessories;
	
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
	
	double* d_ActualEventValue;
	double* d_NextEventValue;
	int*    d_EventCounter;
	int*    d_EventEquilibriumCounter;
	
	double InitialTimeStep;
	int ActiveThreads;
	
	int    DenseOutputNumberOfPoints;
	double DenseOutputTimeStep;
	
	int*    d_DenseOutputIndex;
	double* d_DenseOutputTimeInstances;
	double* d_DenseOutputStates;
	
	int    MaximumNumberOfTimeSteps;
};

template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
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
		int SizeOfIntegerSharedParameters;
		int SizeOfIntegerAccessories;
		
		int SizeOfDenseOutputIndex;
		int SizeOfDenseOutputTimeInstances;
		int SizeOfDenseOutputStates;
		
		size_t DynamicSharedMemory;
		
		double  h_BT_RK4[1];
		double  h_BT_RKCK45[26];
		
		double* h_TimeDomain;
		double* h_ActualState;
		double* h_ControlParameters;
		double* h_SharedParameters;
		double* h_Accessories;
		int*    h_IntegerSharedParameters;
		int*    h_IntegerAccessories;
		
		int*    h_DenseOutputIndex;
		double* h_DenseOutputTimeInstances;
		double* h_DenseOutputStates;
		
		int GridSize;
		int BlockSize;
		
		IntegratorInternalVariables KernelParameters;
		
		void ErrorHandlingSetGetHost(std::string, std::string, int, int);
		
	public:
		ProblemSolver(int);
		~ProblemSolver();
		
		void SetHost(int, VariableSelection, int, double);      // Problem scope, double
		void SetHost(int, IntegerVariableSelection, int, int);  // Problem scope, int
		void SetHost(int, VariableSelection, int, int, double); // Dense state
		void SetHost(VariableSelection, int, double);           // Global scope, double
		void SetHost(IntegerVariableSelection, int, int);       // Global scope, int
		
		void SynchroniseFromHostToDevice(VariableSelection);
		void SynchroniseFromHostToDevice(IntegerVariableSelection);
		void SynchroniseFromDeviceToHost(VariableSelection);
		void SynchroniseFromDeviceToHost(IntegerVariableSelection);
		
		double GetHost(int, VariableSelection, int);            // Problem scope, double
		int    GetHost(int, IntegerVariableSelection, int);     // Problem scope, int
		double GetHost(int, VariableSelection, int, int);       // Dense state
		double GetHost(VariableSelection, int);                 // Global scope, double
		int    GetHost(IntegerVariableSelection, int);          // Global scope, int
		
		void Print(VariableSelection);
		void Print(IntegerVariableSelection);
		void Print(VariableSelection, int);
		
		void SolverOption(ListOfSolverOptions, int);            // int
		void SolverOption(ListOfSolverOptions, double);         // double
		void SolverOption(ListOfSolverOptions, int, int);       // Array of int
		void SolverOption(ListOfSolverOptions, int, double);    // Array of double
		
		void Solve();
		void SynchroniseDevice();
		void InsertSynchronisationPoint();
		void SynchroniseSolver();
};


// --- INCLUDE SOLVERS ---

#include "SingleSystem_PerThread_RungeKutta.cuh"


// --- CUDA DEVICE FUNCTIONS ---

void ListCUDADevices()
{
	int NumberOfDevices;
	cudaGetDeviceCount(&NumberOfDevices);
	for (int i = 0; i < NumberOfDevices; i++)
	{
		cudaDeviceProp CurrentDeviceProperties;
		cudaGetDeviceProperties(&CurrentDeviceProperties, i);
		std::cout << std::endl;
		std::cout << "Device number: " << i << std::endl;
		std::cout << "Device name:   " << CurrentDeviceProperties.name << std::endl;
		std::cout << "--------------------------------------" << std::endl;
		std::cout << "Total global memory:        " << CurrentDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
		std::cout << "Total shared memory:        " << CurrentDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
		std::cout << "Number of 32-bit registers: " << CurrentDeviceProperties.regsPerBlock << std::endl;
		std::cout << "Total constant memory:      " << CurrentDeviceProperties.totalConstMem / 1024 << " Kb" << std::endl;
		std::cout << std::endl;
		std::cout << "Number of multiprocessors:  " << CurrentDeviceProperties.multiProcessorCount << std::endl;
		std::cout << "Compute capability:         " << CurrentDeviceProperties.major << "." << CurrentDeviceProperties.minor << std::endl;
		std::cout << "Core clock rate:            " << CurrentDeviceProperties.clockRate / 1000 << " MHz" << std::endl;
		std::cout << "Memory clock rate:          " << CurrentDeviceProperties.memoryClockRate / 1000 << " MHz" << std::endl;
		std::cout << "Memory bus width:           " << CurrentDeviceProperties.memoryBusWidth  << " bits" << std::endl;
		std::cout << "Peak memory bandwidth:      " << 2.0*CurrentDeviceProperties.memoryClockRate*(CurrentDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
		std::cout << std::endl;
		std::cout << "Warp size:                  " << CurrentDeviceProperties.warpSize << std::endl;
		std::cout << "Max. warps per multiproc:   " << CurrentDeviceProperties.maxThreadsPerMultiProcessor / CurrentDeviceProperties.warpSize << std::endl;
		std::cout << "Max. threads per multiproc: " << CurrentDeviceProperties.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "Max. threads per block:     " << CurrentDeviceProperties.maxThreadsPerBlock << std::endl;
		std::cout << "Max. block dimensions:      " << CurrentDeviceProperties.maxThreadsDim[0] << " * " << CurrentDeviceProperties.maxThreadsDim[1] << " * " << CurrentDeviceProperties.maxThreadsDim[2] << std::endl;
		std::cout << "Max. grid dimensions:       " << CurrentDeviceProperties.maxGridSize[0] << " * " << CurrentDeviceProperties.maxGridSize[1] << " * " << CurrentDeviceProperties.maxGridSize[2] << std::endl;
		std::cout << std::endl;
		std::cout << "Concurrent memory copy:     " << CurrentDeviceProperties.deviceOverlap << std::endl;
		std::cout << "Execution multiple kernels: " << CurrentDeviceProperties.concurrentKernels << std::endl;
		std::cout << "ECC support turned on:      " << CurrentDeviceProperties.ECCEnabled << std::endl << std::endl;
	}
	std::cout << std::endl;
}

int SelectDeviceByClosestRevision(int MajorRevision, int MinorRevision)
{
	int SelectedDevice;
	
	cudaDeviceProp SelectedDeviceProperties;
	memset( &SelectedDeviceProperties, 0, sizeof(cudaDeviceProp) );
	
	SelectedDeviceProperties.major = MajorRevision;
	SelectedDeviceProperties.minor = MinorRevision;
		cudaChooseDevice( &SelectedDevice, &SelectedDeviceProperties );
	
	std::cout << "CUDA Device Number Closest to Revision " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << ": " << SelectedDevice << std::endl << std::endl << std::endl;
	
	return SelectedDevice;
}

void PrintPropertiesOfSpecificDevice(int SelectedDevice)
{
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, SelectedDevice);
	
	std::cout << "Selected device number: " << SelectedDevice << std::endl;
	std::cout << "Selected device name:   " << SelectedDeviceProperties.name << std::endl;
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Total global memory:        " << SelectedDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
	std::cout << "Total shared memory:        " << SelectedDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
	std::cout << "Number of 32-bit registers: " << SelectedDeviceProperties.regsPerBlock << std::endl;
	std::cout << "Total constant memory:      " << SelectedDeviceProperties.totalConstMem / 1024 << " Kb" << std::endl;
	std::cout << std::endl;
	std::cout << "Number of multiprocessors:  " << SelectedDeviceProperties.multiProcessorCount << std::endl;
	std::cout << "Compute capability:         " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << std::endl;
	std::cout << "Core clock rate:            " << SelectedDeviceProperties.clockRate / 1000 << " MHz" << std::endl;
	std::cout << "Memory clock rate:          " << SelectedDeviceProperties.memoryClockRate / 1000 << " MHz" << std::endl;
	std::cout << "Memory bus width:           " << SelectedDeviceProperties.memoryBusWidth  << " bits" << std::endl;
	std::cout << "Peak memory bandwidth:      " << 2.0*SelectedDeviceProperties.memoryClockRate*(SelectedDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
	std::cout << std::endl;
	std::cout << "Warp size:                  " << SelectedDeviceProperties.warpSize << std::endl;
	std::cout << "Max. warps per multiproc:   " << SelectedDeviceProperties.maxThreadsPerMultiProcessor / SelectedDeviceProperties.warpSize << std::endl;
	std::cout << "Max. threads per multiproc: " << SelectedDeviceProperties.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Max. threads per block:     " << SelectedDeviceProperties.maxThreadsPerBlock << std::endl;
	std::cout << "Max. block dimensions:      " << SelectedDeviceProperties.maxThreadsDim[0] << " * " << SelectedDeviceProperties.maxThreadsDim[1] << " * " << SelectedDeviceProperties.maxThreadsDim[2] << std::endl;
	std::cout << "Max. grid dimensions:       " << SelectedDeviceProperties.maxGridSize[0] << " * " << SelectedDeviceProperties.maxGridSize[1] << " * " << SelectedDeviceProperties.maxGridSize[2] << std::endl;
	std::cout << std::endl;
	std::cout << "Concurrent memory copy:     " << SelectedDeviceProperties.deviceOverlap << std::endl;
	std::cout << "Execution multiple kernels: " << SelectedDeviceProperties.concurrentKernels << std::endl;
	std::cout << "ECC support turned on:      " << SelectedDeviceProperties.ECCEnabled << std::endl << std::endl;
	
	std::cout << std::endl;
}

// --- PROBLEM SOLVER OBJECT ---

// CONSTRUCTOR
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::ProblemSolver(int AssociatedDevice)
{
    std::cout << "Creating a SolverObject ..." << std::endl;
	
	// Setup CUDA
	Device = AssociatedDevice;
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
	gpuErrCHK( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
	
	gpuErrCHK( cudaStreamCreate(&Stream) );
	gpuErrCHK( cudaEventCreate(&Event) );
	
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, AssociatedDevice);
	
	
	// Size related user-given variables
	KernelParameters.NumberOfThreads                 = NT;
	KernelParameters.SystemDimension                 = SD;
	KernelParameters.NumberOfControlParameters       = NCP;
	KernelParameters.NumberOfSharedParameters        = NSP;
	KernelParameters.NumberOfEvents                  = NE;
	KernelParameters.NumberOfAccessories             = NA;
	KernelParameters.NumberOfIntegerSharedParameters = NISP;
	KernelParameters.NumberOfIntegerAccessories      = NIA;
	KernelParameters.DenseOutputNumberOfPoints       = NDO;
	
	
	// Global memory management
	SizeOfTimeDomain              = KernelParameters.NumberOfThreads * 2;
	SizeOfActualState             = KernelParameters.NumberOfThreads * KernelParameters.SystemDimension;
	SizeOfControlParameters       = KernelParameters.NumberOfThreads * KernelParameters.NumberOfControlParameters;
	SizeOfSharedParameters        = KernelParameters.NumberOfSharedParameters;
	SizeOfAccessories             = KernelParameters.NumberOfThreads * KernelParameters.NumberOfAccessories;
	SizeOfEvents                  = KernelParameters.NumberOfThreads * KernelParameters.NumberOfEvents;
	SizeOfIntegerSharedParameters = KernelParameters.NumberOfThreads * KernelParameters.NumberOfIntegerSharedParameters;
	SizeOfIntegerAccessories      = KernelParameters.NumberOfThreads * KernelParameters.NumberOfIntegerAccessories;
	
	
	SizeOfDenseOutputIndex         = KernelParameters.NumberOfThreads;
	SizeOfDenseOutputTimeInstances = KernelParameters.NumberOfThreads * KernelParameters.DenseOutputNumberOfPoints;
	SizeOfDenseOutputStates        = KernelParameters.NumberOfThreads * KernelParameters.SystemDimension * KernelParameters.DenseOutputNumberOfPoints;
	
	GlobalMemoryRequired = sizeof(double) * ( SizeOfTimeDomain + 10*SizeOfActualState + SizeOfControlParameters + SizeOfSharedParameters + SizeOfAccessories + \
                                              2*SizeOfEvents + 2*KernelParameters.SystemDimension + KernelParameters.NumberOfEvents + \
											  SizeOfDenseOutputTimeInstances + SizeOfDenseOutputStates ) + \
						   sizeof(int) * ( 2*SizeOfEvents + 2*KernelParameters.NumberOfEvents + SizeOfDenseOutputIndex + SizeOfIntegerSharedParameters + SizeOfIntegerAccessories);
	
	cudaMemGetInfo( &GlobalMemoryFree, &GlobalMemoryTotal );
	std::cout << "   Required global memory:       " << GlobalMemoryRequired/1024/1024 << " Mb" << std::endl;
	std::cout << "   Available free global memory: " << GlobalMemoryFree/1024/1024     << " Mb" << std::endl;
	std::cout << "   Total global memory:          " << GlobalMemoryTotal/1024/1024    << " Mb" << std::endl;
	
	if ( GlobalMemoryRequired >= GlobalMemoryFree )
	{
        std::cerr << "   ERROR: the required amount of global memory is larger than the free!" << std::endl;
		std::cerr << "          Try to reduce the number of points of the DenseOutput or reduce the NumberOfThreads!" << std::endl;
        exit(EXIT_FAILURE);
    }
	std::cout << std::endl;
	
	
	// Shared memory management
	DynamicSharedMemory =  KernelParameters.NumberOfSharedParameters*sizeof(double) + KernelParameters.NumberOfIntegerSharedParameters*sizeof(int);
	DynamicSharedMemory += KernelParameters.NumberOfEvents*( sizeof(int) + sizeof(double) + sizeof(int) );
	
	switch (Algorithm)
	{
		case RKCK45:
			DynamicSharedMemory += 2*KernelParameters.SystemDimension*sizeof(double);
			break;
	}
	
	std::cout << "   Total shared memory required:  " << DynamicSharedMemory                        << " b" << std::endl;
	std::cout << "   Total shared memory available: " << SelectedDeviceProperties.sharedMemPerBlock << " b" << std::endl;
	if ( DynamicSharedMemory >= SelectedDeviceProperties.sharedMemPerBlock )
	{
        std::cerr << "   ERROR: the required amount of shared memory is larger than the free!" << std::endl;
		std::cerr << "          Try to reduce the number the SharedParameters!" << std::endl;
        exit(EXIT_FAILURE);
    }
	
	
	// Constant memory management		
	h_BT_RK4[0] = 1.0/6.0;
	
	h_BT_RKCK45[0]  =     1.0/5.0;
	h_BT_RKCK45[1]  =     3.0/10.0;
	h_BT_RKCK45[2]	=     3.0/40.0;
	h_BT_RKCK45[3]  =     9.0/40.0;
	h_BT_RKCK45[4]  =     3.0/5.0;
	h_BT_RKCK45[5]  =    -9.0/10.0;
	h_BT_RKCK45[6]  =     6.0/5.0;
	h_BT_RKCK45[7]  =   -11.0/54.0;
	h_BT_RKCK45[8]  =     5.0/2.0;
	h_BT_RKCK45[9]  =   -70.0/27.0;
	h_BT_RKCK45[10] =    35.0/27.0;
	h_BT_RKCK45[11] =     7.0/8.0;
	h_BT_RKCK45[12] =  1631.0/55296.0;
	h_BT_RKCK45[13] =   175.0/512.0;
	h_BT_RKCK45[14] =   575.0/13824.0;
	h_BT_RKCK45[15] = 44275.0/110592.0;
	h_BT_RKCK45[16] =   253.0/4096.0;
	h_BT_RKCK45[17] =    37.0/378.0;
	h_BT_RKCK45[18] =   250.0/621.0;
	h_BT_RKCK45[19] =   125.0/594.0;
	h_BT_RKCK45[20] =   512.0/1771.0;
	h_BT_RKCK45[21] =  2825.0/27648.0;
	h_BT_RKCK45[22] = 18575.0/48384.0;
	h_BT_RKCK45[23] = 13525.0/55296.0;
	h_BT_RKCK45[24] =   277.0/14336.0;
	h_BT_RKCK45[25] =     1.0/4.0;
	
	gpuErrCHK( cudaMemcpyToSymbol(d_BT_RK4,    h_BT_RK4,     1*sizeof(double)) );
	gpuErrCHK( cudaMemcpyToSymbol(d_BT_RKCK45, h_BT_RKCK45, 26*sizeof(double)) );
	
	
	// Host internal variables
	h_TimeDomain               = AllocateHostPinnedMemory<double>( SizeOfTimeDomain );
	h_ActualState              = AllocateHostPinnedMemory<double>( SizeOfActualState );
	h_ControlParameters        = AllocateHostPinnedMemory<double>( SizeOfControlParameters );
	h_SharedParameters         = AllocateHostPinnedMemory<double>( SizeOfSharedParameters );
	h_Accessories              = AllocateHostPinnedMemory<double>( SizeOfAccessories );
	h_IntegerSharedParameters  = AllocateHostPinnedMemory<int>( SizeOfIntegerSharedParameters );
	h_IntegerAccessories       = AllocateHostPinnedMemory<int>( SizeOfIntegerAccessories );
	h_DenseOutputIndex         = AllocateHostPinnedMemory<int>( SizeOfDenseOutputIndex );
	h_DenseOutputTimeInstances = AllocateHostPinnedMemory<double>( SizeOfDenseOutputTimeInstances );
	h_DenseOutputStates        = AllocateHostPinnedMemory<double>( SizeOfDenseOutputStates );
	
	
	// Device internal variables
	KernelParameters.d_TimeDomain              = AllocateDeviceMemory<double>( SizeOfTimeDomain );
	KernelParameters.d_ActualState             = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_ControlParameters       = AllocateDeviceMemory<double>( SizeOfControlParameters );
	KernelParameters.d_SharedParameters        = AllocateDeviceMemory<double>( SizeOfSharedParameters );
	KernelParameters.d_Accessories             = AllocateDeviceMemory<double>( SizeOfAccessories );
	KernelParameters.d_IntegerSharedParameters = AllocateDeviceMemory<int>( SizeOfIntegerSharedParameters );
	KernelParameters.d_IntegerAccessories      = AllocateDeviceMemory<int>( SizeOfIntegerAccessories );
	
	KernelParameters.d_State    = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_Stages   = AllocateDeviceMemory<double>( SizeOfActualState * 6 );
	
	KernelParameters.d_NextState = AllocateDeviceMemory<double>( SizeOfActualState );
	
	KernelParameters.d_Error           = AllocateDeviceMemory<double>( SizeOfActualState );
	
	KernelParameters.d_ActualEventValue        = AllocateDeviceMemory<double>( SizeOfEvents );
	KernelParameters.d_NextEventValue          = AllocateDeviceMemory<double>( SizeOfEvents );
	KernelParameters.d_EventCounter            = AllocateDeviceMemory<int>( SizeOfEvents );
	KernelParameters.d_EventEquilibriumCounter = AllocateDeviceMemory<int>( SizeOfEvents );
	
	KernelParameters.d_RelativeTolerance       = AllocateDeviceMemory<double>( KernelParameters.SystemDimension );
	KernelParameters.d_AbsoluteTolerance       = AllocateDeviceMemory<double>( KernelParameters.SystemDimension );
	KernelParameters.d_EventTolerance          = AllocateDeviceMemory<double>( KernelParameters.NumberOfEvents );
	KernelParameters.d_EventDirection          = AllocateDeviceMemory<int>( KernelParameters.NumberOfEvents );
	KernelParameters.d_EventStopCounter        = AllocateDeviceMemory<int>( KernelParameters.NumberOfEvents );
	
	KernelParameters.d_DenseOutputIndex         = AllocateDeviceMemory<int>( SizeOfDenseOutputIndex );
	KernelParameters.d_DenseOutputTimeInstances = AllocateDeviceMemory<double>( SizeOfDenseOutputTimeInstances );
	KernelParameters.d_DenseOutputStates        = AllocateDeviceMemory<double>( SizeOfDenseOutputStates );
	
	KernelParameters.InitialTimeStep = 1e-2;
	KernelParameters.ActiveThreads   = KernelParameters.NumberOfThreads;
	
	KernelParameters.MaximumTimeStep     = 1.0e6;
	KernelParameters.MinimumTimeStep     = 1.0e-12;
	KernelParameters.TimeStepGrowLimit   = 5.0;
	KernelParameters.TimeStepShrinkLimit = 0.1;
	
	KernelParameters.MaxStepInsideEvent  = 50;
	
	KernelParameters.DenseOutputTimeStep = -1e-2;
	
	KernelParameters.MaximumNumberOfTimeSteps = 0;
	
	
	// Kernel configuration
	BlockSize  = SelectedDeviceProperties.warpSize;
	GridSize   = KernelParameters.NumberOfThreads/BlockSize + (KernelParameters.NumberOfThreads % BlockSize == 0 ? 0:1);
	
	
	// Default integration tolerances
	double DefaultTolerances = 1e-8;
	for (int i=0; i<KernelParameters.SystemDimension; i++)
	{
		gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_RelativeTolerance + i, &DefaultTolerances, sizeof(double), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_AbsoluteTolerance + i, &DefaultTolerances, sizeof(double), cudaMemcpyHostToDevice, Stream) );
	}
	
	DefaultTolerances = 1e-6;
	int EventStopCounterAndDirection = 0;
	for (int i=0; i<KernelParameters.NumberOfEvents; i++)
	{
		gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventDirection   + i, &EventStopCounterAndDirection, sizeof(int), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventStopCounter + i, &EventStopCounterAndDirection, sizeof(int), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventTolerance   + i, &DefaultTolerances,         sizeof(double), cudaMemcpyHostToDevice, Stream) );
	}
	
	std::cout << std::endl << std::endl;
	
	std::cout << "Object for Parameters scan is successfully created! Required memory allocations have been done" << std::endl << std::endl;
}

// DESTRUCTOR
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::~ProblemSolver()
{
    gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaStreamDestroy(Stream) );
	gpuErrCHK( cudaEventDestroy(Event) );
	
	gpuErrCHK( cudaFreeHost(h_TimeDomain) );
	gpuErrCHK( cudaFreeHost(h_ActualState) );
	gpuErrCHK( cudaFreeHost(h_ControlParameters) );
	gpuErrCHK( cudaFreeHost(h_SharedParameters) );
	gpuErrCHK( cudaFreeHost(h_Accessories) );
	gpuErrCHK( cudaFreeHost(h_IntegerSharedParameters) );
	gpuErrCHK( cudaFreeHost(h_IntegerAccessories) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputIndex) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputStates) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_TimeDomain) );
	gpuErrCHK( cudaFree(KernelParameters.d_ActualState) );
	gpuErrCHK( cudaFree(KernelParameters.d_ControlParameters) );
	gpuErrCHK( cudaFree(KernelParameters.d_SharedParameters) );
	gpuErrCHK( cudaFree(KernelParameters.d_Accessories) );
	gpuErrCHK( cudaFree(KernelParameters.d_IntegerSharedParameters) );
	gpuErrCHK( cudaFree(KernelParameters.d_IntegerAccessories) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_State) );
	gpuErrCHK( cudaFree(KernelParameters.d_Stages) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_NextState) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_Error) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_ActualEventValue) );
	gpuErrCHK( cudaFree(KernelParameters.d_NextEventValue) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventCounter) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventEquilibriumCounter) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_RelativeTolerance) );
	gpuErrCHK( cudaFree(KernelParameters.d_AbsoluteTolerance) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventTolerance) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventDirection) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventStopCounter) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_DenseOutputIndex) );
	gpuErrCHK( cudaFree(KernelParameters.d_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFree(KernelParameters.d_DenseOutputStates) );
	
	std::cout << "Object for Parameters scan is deleted! Every memory have been deallocated!" << std::endl << std::endl;
}

// ERROR HANDLING, set/get host, options
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::ErrorHandlingSetGetHost(std::string Function, std::string Variable, int Value, int Limit)
{
	if ( Value >= Limit )
	{
        std::cerr << "ERROR in solver member function " << Function << ":"  << std::endl << "    "\
		          << "The index of " << Variable << " cannot be larger than " << Limit-1   << "! "\
			      << "(The indexing starts from zero)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// SETHOST, Problem scope, double
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber, double Value)
{
	ErrorHandlingSetGetHost("SetHost", "ProblemNumber", ProblemNumber, KernelParameters.NumberOfThreads);
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	switch (Variable)
	{
		case TimeDomain:
			ErrorHandlingSetGetHost("SetHost", "TimeDomain", SerialNumber, 2);
			h_TimeDomain[idx] = Value;
			break;
		
		case ActualState:
			ErrorHandlingSetGetHost("SetHost", "ActualState", SerialNumber, KernelParameters.SystemDimension);
			h_ActualState[idx] = Value;
			break;
		
		case ControlParameters:
			ErrorHandlingSetGetHost("SetHost", "ControlParameters", SerialNumber, KernelParameters.NumberOfControlParameters);
			h_ControlParameters[idx] = Value;
			break;
		
		case Accessories:
			ErrorHandlingSetGetHost("SetHost", "Accessories", SerialNumber, KernelParameters.NumberOfAccessories);
			h_Accessories[idx] = Value;
			break;
		
		case DenseTime:
			ErrorHandlingSetGetHost("SetHost", "DenseTime", SerialNumber, KernelParameters.DenseOutputNumberOfPoints);
			h_DenseOutputTimeInstances[idx] = Value;
			break;
		
		default:
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (double instead of int)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Problem scope, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, IntegerVariableSelection Variable, int SerialNumber, int Value)
{
	ErrorHandlingSetGetHost("SetHost", "ProblemNumber", ProblemNumber, KernelParameters.NumberOfThreads);
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	switch (Variable)
	{
		case IntegerAccessories:
			ErrorHandlingSetGetHost("SetHost", "IntegerAccessories", SerialNumber, KernelParameters.NumberOfIntegerAccessories);
			h_IntegerAccessories[idx] = Value;
			break;
		
		default:
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (int instead of double)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Dense state
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber, int TimeStep, double Value)
{
	ErrorHandlingSetGetHost("SetHost", "ProblemNumber", ProblemNumber, KernelParameters.NumberOfThreads);
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads + TimeStep*KernelParameters.NumberOfThreads*KernelParameters.SystemDimension;
	
	switch (Variable)
	{
		case DenseState:
			ErrorHandlingSetGetHost("SetHost", "DenseState", SerialNumber, KernelParameters.SystemDimension);
			ErrorHandlingSetGetHost("SetHost", "DenseState/TimeStep", TimeStep, KernelParameters.DenseOutputNumberOfPoints);
			h_DenseOutputStates[idx] = Value;
			break;
		
		default:
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Global scope, double
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(VariableSelection Variable, int SerialNumber, double Value)
{
	switch (Variable)
	{
		case SharedParameters:
			ErrorHandlingSetGetHost("SetHost", "SharedParameters", SerialNumber, KernelParameters.NumberOfSharedParameters);
			h_SharedParameters[SerialNumber] = Value;
			break;
		
		default:
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (int instead of double)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Global scope, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(IntegerVariableSelection Variable, int SerialNumber, int Value)
{
	switch (Variable)
	{
		case IntegerSharedParameters:
			ErrorHandlingSetGetHost("SetHost", "IntegerSharedParameters", SerialNumber, KernelParameters.NumberOfIntegerSharedParameters);
			h_IntegerSharedParameters[SerialNumber] = Value;
			break;
		
		default:
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (double instead of int)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, H->D, default
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromHostToDevice(VariableSelection Variable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (Variable)
	{
		case TimeDomain:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_TimeDomain, h_TimeDomain, SizeOfTimeDomain*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case ActualState:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ActualState, h_ActualState, SizeOfActualState*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case ControlParameters:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ControlParameters, h_ControlParameters, SizeOfControlParameters*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case SharedParameters:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_SharedParameters, h_SharedParameters, SizeOfSharedParameters*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case Accessories:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_Accessories, h_Accessories, SizeOfAccessories*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case DenseOutput:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputIndex, h_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputTimeInstances, h_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputStates, h_DenseOutputStates, SizeOfDenseOutputStates*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case All:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_TimeDomain, h_TimeDomain, SizeOfTimeDomain*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ActualState, h_ActualState, SizeOfActualState*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ControlParameters, h_ControlParameters, SizeOfControlParameters*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_SharedParameters, h_SharedParameters, SizeOfSharedParameters*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_Accessories, h_Accessories, SizeOfAccessories*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_IntegerSharedParameters, h_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_IntegerAccessories, h_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputIndex, h_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputTimeInstances, h_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputStates, h_DenseOutputStates, SizeOfDenseOutputStates*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		default:
			std::cerr << "ERROR in solver member function SynchroniseFromHostToDevice:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, H->D, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromHostToDevice(IntegerVariableSelection Variable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (Variable)
	{
		case IntegerSharedParameters:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_IntegerSharedParameters, h_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case IntegerAccessories:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_IntegerAccessories, h_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
			
		default:
			std::cerr << "ERROR in solver member function SynchroniseFromHostToDevice:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, D->H, default
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromDeviceToHost(VariableSelection Variable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (Variable)
	{
		case TimeDomain:
			gpuErrCHK( cudaMemcpyAsync(h_TimeDomain, KernelParameters.d_TimeDomain, SizeOfTimeDomain*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case ActualState:
			gpuErrCHK( cudaMemcpyAsync(h_ActualState, KernelParameters.d_ActualState, SizeOfActualState*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case ControlParameters:
			gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, KernelParameters.d_ControlParameters, SizeOfControlParameters*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case SharedParameters:
			gpuErrCHK( cudaMemcpyAsync(h_SharedParameters, KernelParameters.d_SharedParameters, SizeOfSharedParameters*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case Accessories:
			gpuErrCHK( cudaMemcpyAsync(h_Accessories, KernelParameters.d_Accessories, SizeOfAccessories*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case DenseOutput:
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex, KernelParameters.d_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, KernelParameters.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates, KernelParameters.d_DenseOutputStates, SizeOfDenseOutputStates*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case All:
			gpuErrCHK( cudaMemcpyAsync(h_TimeDomain, KernelParameters.d_TimeDomain, SizeOfTimeDomain*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ActualState, KernelParameters.d_ActualState, SizeOfActualState*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, KernelParameters.d_ControlParameters, SizeOfControlParameters*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_SharedParameters, KernelParameters.d_SharedParameters, SizeOfSharedParameters*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_Accessories, KernelParameters.d_Accessories, SizeOfAccessories*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerSharedParameters, KernelParameters.d_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerAccessories, KernelParameters.d_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex, KernelParameters.d_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, KernelParameters.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates, KernelParameters.d_DenseOutputStates, SizeOfDenseOutputStates*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		default:
			std::cerr << "ERROR in solver member function SynchroniseFromDeviceToHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, D->H, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromDeviceToHost(IntegerVariableSelection Variable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (Variable)
	{
		case IntegerSharedParameters:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerSharedParameters, KernelParameters.d_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case IntegerAccessories:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerAccessories, KernelParameters.d_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		default:
			std::cerr << "ERROR in solver member function SynchroniseFromDeviceToHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Problem scope, double
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
double ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber)
{
	ErrorHandlingSetGetHost("GetHost", "ProblemNumber", ProblemNumber, KernelParameters.NumberOfThreads);
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	double Value;
	switch (Variable)
	{
		case TimeDomain:
			ErrorHandlingSetGetHost("GetHost", "TimeDomain", SerialNumber, 2);
			Value = h_TimeDomain[idx];
			break;
			
		case ActualState:
			ErrorHandlingSetGetHost("GetHost", "ActualState", SerialNumber, KernelParameters.SystemDimension);
			Value = h_ActualState[idx];
			break;
			
		case ControlParameters:
			ErrorHandlingSetGetHost("GetHost", "ControlParameters", SerialNumber, KernelParameters.NumberOfControlParameters);
			Value = h_ControlParameters[idx];
			break;
			
		case Accessories:
			ErrorHandlingSetGetHost("GetHost", "Accessories", SerialNumber, KernelParameters.NumberOfAccessories);
			Value = h_Accessories[idx];
			break;
			
		case DenseTime:
			ErrorHandlingSetGetHost("GetHost", "DenseTime", SerialNumber, KernelParameters.DenseOutputNumberOfPoints);
			Value = h_DenseOutputTimeInstances[idx];
			break;
			
		default:
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// GETHOST, Problem scope, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
int ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, IntegerVariableSelection Variable, int SerialNumber)
{
	ErrorHandlingSetGetHost("GetHost", "ProblemNumber", ProblemNumber, KernelParameters.NumberOfThreads);
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	double Value;
	switch (Variable)
	{
		case IntegerAccessories:
			ErrorHandlingSetGetHost("GetHost", "IntegerAccessories", SerialNumber, KernelParameters.NumberOfIntegerAccessories);
			Value = h_IntegerAccessories[idx];
			break;
			
		default:
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// GETHOST, Dense state
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
double ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber, int TimeStep)
{
	ErrorHandlingSetGetHost("GetHost", "ProblemNumber", ProblemNumber, KernelParameters.NumberOfThreads);
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads + TimeStep*KernelParameters.NumberOfThreads*KernelParameters.SystemDimension;
	
	double Value;
	switch (Variable)
	{
		case DenseState:
			ErrorHandlingSetGetHost("GetHost", "DenseState", SerialNumber, KernelParameters.SystemDimension);
			ErrorHandlingSetGetHost("GetHost", "DenseState/TimeStep", TimeStep, KernelParameters.DenseOutputNumberOfPoints);
			Value = h_DenseOutputStates[idx];
			break;
		
		default :
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// GETHOST, Global scope, double
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
double ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(VariableSelection Variable, int SerialNumber)
{
	double Value;
	switch (Variable)
	{
		case SharedParameters:
			ErrorHandlingSetGetHost("GetHost", "SharedParameters", SerialNumber, KernelParameters.NumberOfSharedParameters);
			Value = h_SharedParameters[SerialNumber];
			break;
		
		default :
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// GETHOST, Global scope, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
int ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(IntegerVariableSelection Variable, int SerialNumber)
{
	double Value;
	switch (Variable)
	{
		case IntegerSharedParameters:
			ErrorHandlingSetGetHost("GetHost", "IntegerSharedParameters", SerialNumber, KernelParameters.NumberOfIntegerSharedParameters);
			Value = h_IntegerSharedParameters[SerialNumber];
			break;
		
		default :
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// PRINT, default
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(VariableSelection Variable)
{
	std::ofstream DataFile;
	int NumberOfRows;
	int NumberOfColumns;
	double* PointerToActualData;
	
	switch (Variable)
	{
		case TimeDomain:
			DataFile.open ( "TimeDomainInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfThreads;
			NumberOfColumns = 2;
			PointerToActualData = h_TimeDomain;
			break;
			
		case ActualState:
			DataFile.open ( "ActualStateInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfThreads;
			NumberOfColumns = KernelParameters.SystemDimension;
			PointerToActualData = h_ActualState;
			break;
			
		case ControlParameters:
			DataFile.open ( "ControlParametersInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfThreads;
			NumberOfColumns = KernelParameters.NumberOfControlParameters;
			PointerToActualData = h_ControlParameters;
			break;
			
		case SharedParameters:
			DataFile.open ( "SharedParametersInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfSharedParameters;		
			NumberOfColumns = 1;
			PointerToActualData = h_SharedParameters;
			break;
			
		case Accessories:
			DataFile.open ( "AccessoriesInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfThreads;
			NumberOfColumns = KernelParameters.NumberOfAccessories;
			PointerToActualData = h_Accessories;
			break;
		
		default :
			std::cerr << "ERROR in solver member function Print:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int idx;
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			idx = i + j*NumberOfRows;
			DataFile.width(Width); DataFile << PointerToActualData[idx];
			if ( j<(NumberOfColumns-1) )
				DataFile << ',';
		}
		DataFile << '\n';
	}
	
	DataFile.close();
}

// PRINT, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(IntegerVariableSelection Variable)
{
	std::ofstream DataFile;
	int NumberOfRows;
	int NumberOfColumns;
	int* PointerToActualData;
	
	switch (Variable)
	{
		case IntegerSharedParameters:
			DataFile.open ( "IntegerSharedParametersInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfIntegerSharedParameters;		
			NumberOfColumns = 1;
			PointerToActualData = h_IntegerSharedParameters;
			break;
			
		case IntegerAccessories:
			DataFile.open ( "IntegerAccessoriesInSolverObject.txt" );
			NumberOfRows = KernelParameters.NumberOfThreads;
			NumberOfColumns = KernelParameters.NumberOfIntegerAccessories;
			PointerToActualData = h_IntegerAccessories;
			break;
			
		default :
			std::cerr << "ERROR in solver member function Print:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int idx;
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			idx = i + j*NumberOfRows;
			DataFile.width(Width); DataFile << PointerToActualData[idx];
			if ( j<(NumberOfColumns-1) )
				DataFile << ',';
		}
		DataFile << '\n';
	}
	
	DataFile.close();
}

// PRINT, Dense state
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(VariableSelection Variable, int ThreadID)
{
	ErrorHandlingSetGetHost("Print", "Thread", ThreadID, KernelParameters.NumberOfThreads);
	
	if ( Variable != DenseOutput )
	{
        std::cerr << "ERROR in solver member function Print:" << std::endl << "    "\
			      << "Invalid option for variable selection!" << std::endl;
        exit(EXIT_FAILURE);
    }
	
	std::ofstream DataFile;
	std::string FileName = "tid_" + std::to_string(ThreadID) + ".txt";
	DataFile.open ( FileName.c_str() );
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int idx;
	DataFile << "ControlParameters:\n";
	for (int i=0; i<KernelParameters.NumberOfControlParameters; i++)
	{
		idx = ThreadID + i*KernelParameters.NumberOfThreads;
		DataFile.width(Width); DataFile << h_ControlParameters[idx];
		if ( i<(KernelParameters.NumberOfControlParameters-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "SharedParameters:\n";
	for (int i=0; i<KernelParameters.NumberOfSharedParameters; i++)
	{
		DataFile.width(Width); DataFile << h_SharedParameters[i];
		if ( i<(KernelParameters.NumberOfSharedParameters-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "IntegerSharedParameters:\n";
	for (int i=0; i<KernelParameters.NumberOfIntegerSharedParameters; i++)
	{
		DataFile.width(Width); DataFile << h_IntegerSharedParameters[i];
		if ( i<(KernelParameters.NumberOfIntegerSharedParameters-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "Accessories:\n";
	for (int i=0; i<KernelParameters.NumberOfAccessories; i++)
	{
		idx = ThreadID + i*KernelParameters.NumberOfThreads;
		DataFile.width(Width); DataFile << h_Accessories[idx];
		if ( i<(KernelParameters.NumberOfAccessories-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "IntegerAccessories:\n";
	for (int i=0; i<KernelParameters.NumberOfIntegerAccessories; i++)
	{
		idx = ThreadID + i*KernelParameters.NumberOfThreads;
		DataFile.width(Width); DataFile << h_IntegerAccessories[idx];
		if ( i<(KernelParameters.NumberOfIntegerAccessories-1) )
			DataFile << ',';
	}
	DataFile << "\n\n";
	
	DataFile << "Time series:\n";
	if ( NDO > 0 )
	{
		for (int i=0; i<(h_DenseOutputIndex[ThreadID]+1); i++)
		{
			idx = ThreadID + i*KernelParameters.NumberOfThreads;
			DataFile.width(Width); DataFile << h_DenseOutputTimeInstances[idx] << ',';
			
			for (int j=0; j<KernelParameters.SystemDimension; j++)
			{
				idx = ThreadID + j*KernelParameters.NumberOfThreads + i*KernelParameters.NumberOfThreads*KernelParameters.SystemDimension;
				DataFile.width(Width); DataFile << h_DenseOutputStates[idx];
				if ( j<(KernelParameters.SystemDimension-1) )
					DataFile << ',';
			}
			DataFile << '\n';
		}
	}
	
	DataFile.close();
}

// OPTION, int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, int Value)
{
	switch (Option)
	{
		case ThreadsPerBlock:
			BlockSize = Value;
			GridSize = KernelParameters.NumberOfThreads/BlockSize + (KernelParameters.NumberOfThreads % BlockSize == 0 ? 0:1);
			break;
		
		case ActiveNumberOfThreads:
			KernelParameters.ActiveThreads = Value;
			break;
		
		case MaxStepInsideEvent:
			KernelParameters.MaxStepInsideEvent = Value;
			break;
		
		case MaximumNumberOfTimeSteps:
			KernelParameters.MaximumNumberOfTimeSteps = Value;
			break;
			
		default:
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
					  << "Invalid option for variable selection or wrong type of input value (expected data type is int)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// OPTION, double
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, double Value)
{
	switch (Option)
	{
		case InitialTimeStep:
			KernelParameters.InitialTimeStep = Value;
			break;
		
		case MaximumTimeStep:
			KernelParameters.MaximumTimeStep = Value;
			break;
		
		case MinimumTimeStep:
			KernelParameters.MinimumTimeStep = Value;
			break;
		
		case TimeStepGrowLimit:
			KernelParameters.TimeStepGrowLimit = Value;
			break;
		
		case TimeStepShrinkLimit:
			KernelParameters.TimeStepShrinkLimit = Value;
			break;
		
		case DenseOutputTimeStep:
			KernelParameters.DenseOutputTimeStep = Value;
			break;
			
		default:
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (expected data type is double)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// OPTION, array of int
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, int SerialNumber, int Value)
{
	switch (Option)
	{
		case EventDirection:
			ErrorHandlingSetGetHost("SolverOption", "EventDirection", SerialNumber, KernelParameters.NumberOfEvents);
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventDirection+SerialNumber, &Value, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case EventStopCounter:
			ErrorHandlingSetGetHost("SolverOption", "EventStopCounter", SerialNumber, KernelParameters.NumberOfEvents);
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventStopCounter+SerialNumber, &Value, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
			
		default:
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (expected data type is int)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// OPTION, array of double
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, int SerialNumber, double Value)
{
	switch (Option)
	{
		case RelativeTolerance:
			ErrorHandlingSetGetHost("SolverOption", "RelativeTolerance", SerialNumber, KernelParameters.SystemDimension);
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_RelativeTolerance+SerialNumber, &Value, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case AbsoluteTolerance:
			ErrorHandlingSetGetHost("SolverOption", "AbsoluteTolerance", SerialNumber, KernelParameters.SystemDimension);
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_AbsoluteTolerance+SerialNumber, &Value, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case EventTolerance:
			ErrorHandlingSetGetHost("SolverOption", "EventTolerance", SerialNumber, KernelParameters.NumberOfEvents);
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventTolerance+SerialNumber, &Value, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		default :
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			          << "Invalid option for variable selection or wrong type of input value (expected data type is double)!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SOLVE
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Solve()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	SingleSystem_PerThread_RungeKutta<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision><<<GridSize, BlockSize, DynamicSharedMemory, Stream>>> (KernelParameters);
}

// SYNCHRONISE DEVICE
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseDevice()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaDeviceSynchronize() );
}

// INSERT SYNCHRONISATION POINT
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::InsertSynchronisationPoint()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaEventRecord(Event, Stream) );
}

// SYNCHRONISE SOLVER
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseSolver()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaEventSynchronize(Event) );
}

// --- AUXILIARY FUNCTIONS ---

template <class DataType>
DataType* AllocateDeviceMemory(int N)
{
    cudaError_t Error = cudaSuccess;
	
	DataType* MemoryAddressInDevice = NULL;
	
	Error = cudaMalloc((void**)&MemoryAddressInDevice, N * sizeof(DataType));
    
	if (Error != cudaSuccess)
    {
        std::cerr << "Failed to allocate Memory on the DEVICE!\n";
        exit(EXIT_FAILURE);
    }
    return MemoryAddressInDevice;
}

template <class DataType>
DataType* AllocateHostPinnedMemory(int N)
{
    cudaError_t Error = cudaSuccess;
	
	DataType* MemoryAddressInHost = NULL;
	
	Error = cudaMallocHost((void**)&MemoryAddressInHost, N * sizeof(DataType));
    
	if (Error != cudaSuccess)
    {
        std::cerr << "Failed to allocate Pinned Memory on the HOST!\n";
        exit(EXIT_FAILURE);
    }
    return MemoryAddressInHost;
}

template <class DataType>
DataType* AllocateHostMemory(int N)
{
    DataType* HostMemory = new (std::nothrow) DataType [N];
    if (HostMemory == NULL)
    {
        std::cerr << "Failed to allocate Memory on the HOST!\n";
        exit(EXIT_FAILURE);
    }
    return HostMemory;
}

#endif
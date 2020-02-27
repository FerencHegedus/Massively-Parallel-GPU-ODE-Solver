#ifndef COUPLEDSYSTEMS_PERBLOCK_H
#define COUPLEDSYSTEMS_PERBLOCK_H

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
enum ListOfSolverOptions{  };


void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);


struct IntegratorInternalVariables
{
	int NumberOfSystems;
	int UnitsPerSystem;
	int UnitDimension;
	int ThreadsPerBlock;
	int SystemPerBlock;
	int NumberOfCouplings;
	int NumberOfUnitParameters;
	int NumberOfSystemParameters ;
	int NumberOfGlobalParameters;
	int NumberOfIntegerGlobalParameters;
	int NumberOfUnitAccessories;
	int NumberOfIntegerUnitAccessories;
	int NumberOfSystemAccessories;
	int NumberOfIntegerSystemAccessories;
	int NumberOfEvents;
	int DenseOutputNumberOfPoints;
	
	int ThreadsPerBlockRequired;
	int NumberOfBlockLaunches;
	int ThreadPadding;
	int GridSize;
	int ThreadAllocationRequired;
	
	
	
	double* d_TimeDomain;
	double* d_ActualState;
	double* d_ControlParameters;
	double* d_SharedParameters;
	double* d_Accessories;
	
	double* d_CouplingMatrix;
	
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
	int ActiveSystems;
};

template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int, NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
class ProblemSolver
{
    private:
		int Device;
		cudaStream_t Stream;
		cudaEvent_t Event;
		
		IntegratorInternalVariables KernelParameters;
		
		int SizeOfTimeDomain;
		int SizeOfActualState;
		int SizeOfControlParameters;
		int SizeOfSharedParameters;
		int SizeOfAccessories;
		int SizeOfEvents;
		int SizeOfCouplingMatrix;
		
		
		
		
		size_t DynamicSharedMemory_RK4_EH0_SSSBL;
		size_t DynamicSharedMemoryRKCK45;
		size_t DynamicSharedMemoryRKCK45_EH0;
		size_t DynamicSharedMemoryRK4;
		
		double  h_BT_RK4[1];
		double  h_BT_RKCK45[26];
		
		double* h_TimeDomain;
		double* h_ActualState;
		double* h_ControlParameters;
		double* h_SharedParameters;
		double* h_Accessories;
		
		double* h_CouplingMatrix;
		

		
		
		
		ListOfSolverAlgorithms Solver;
		
	public:
		ProblemSolver(const ConstructorConfiguration&, int);
		~ProblemSolver();
		
		void SetHost(int, int, VariableSelection, int, double); // Unit scope
		void SetHost(int, VariableSelection, int, double);      // System scope
		void SetHost(VariableSelection, int, double);           // Global scope
		void SetHost(VariableSelection, int, int, double);      // Coupling matrix
		void SynchroniseFromHostToDevice(VariableSelection);
		void SynchroniseFromDeviceToHost(VariableSelection);
		double GetHost(int, int, VariableSelection, int); // Unit scope
		double GetHost(int, VariableSelection, int);      // System scope
		double GetHost(VariableSelection, int);           // Global scope
		double GetHost(VariableSelection, int, int);      // Coupling matrix
		
		void Print(VariableSelection);
		
		void SolverOption(ListOfSolverAlgorithms, double, int);
		void Solve();
		
		void SynchroniseDevice();
		void InsertSynchronisationPoint();
		void SynchroniseSolver();
};


// --- INCLUDE SOLVERS ---




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
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int, NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,double>::ProblemSolver(int AssociatedDevice)
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
	KernelParameters.NumberOfSystems                  = NS;
	KernelParameters.UnitsPerSystem                   = UPS;
	KernelParameters.UnitDimension                    = UD;
	KernelParameters.ThreadsPerBlock                  = TPB;
	KernelParameters.SystemPerBlock                   = SPB;
	KernelParameters.NumberOfCouplings                = NC;
	KernelParameters.NumberOfUnitParameters           = NUP;
	KernelParameters.NumberOfSystemParameters         = NSP;
	KernelParameters.NumberOfGlobalParameters         = NGP;
	KernelParameters.NumberOfIntegerGlobalParameters  = NiGP;
	KernelParameters.NumberOfUnitAccessories          = NUA;
	KernelParameters.NumberOfIntegerUnitAccessories   = NiUA;
	KernelParameters.NumberOfSystemAccessories        = NSA;
	KernelParameters.NumberOfIntegerSystemAccessories = NiSA;
	KernelParameters.NumberOfEvents                   = NiSA;
	KernelParameters.DenseOutputNumberOfPoints        = NDO;
	
	
	// Management of threads, thread padding, blocks, block launches
	KernelParameters.ThreadsPerBlockRequired = KernelParameters.SystemPerBlock * KernelParameters.UnitsPerSystem;
	KernelParameters.NumberOfBlockLaunches   = KernelParameters.ThreadsPerBlockRequired / KernelParameters.ThreadsPerBlock + (KernelParameters.ThreadsPerBlockRequired % KernelParameters.ThreadsPerBlock == 0 ? 0:1);
	KernelParameters.ThreadPadding           = KernelParameters.NumberOfBlockLaunches * KernelParameters.ThreadsPerBlock - KernelParameters.ThreadsPerBlockRequired;
	
	KernelParameters.GridSize = KernelParameters.NumberOfSystems/KernelParameters.SystemPerBlock + (KernelParameters.NumberOfSystems % KernelParameters.SystemPerBlock == 0 ? 0:1);
	KernelParameters.ThreadAllocationRequired = (KernelParameters.ThreadsPerBlockRequired + KernelParameters.ThreadPadding) * KernelParameters.GridSize;
	
	cout << "Total Number of Systems:    " << KernelParameters.NumberOfSystems << endl;
	cout << "System per Blocks:          " << KernelParameters.SystemPerBlock << endl;
	cout << "Threads per Block Required: " << KernelParameters.ThreadsPerBlockRequired << endl;
	cout << "Number of Block Launches:   " << KernelParameters.NumberOfBlockLaunches << endl;
	cout << "Thread Padding:             " << KernelParameters.ThreadPadding << endl;
	cout << "GridSize:                   " << KernelParameters.GridSize << endl << endl;
	
	cout << "Total Number of Threads Required : " << KernelParameters.ThreadAllocationRequired << endl;
	cout << "Thread Efficinecy                : " << (double)(KernelParameters.NumberOfSystems*KernelParameters.UnitsPerSystem)/KernelParameters.ThreadAllocationRequired << endl;
	cout << "Total Number of Idle Threads     : " << KernelParameters.ThreadAllocationRequired - (KernelParameters.NumberOfSystems*KernelParameters.UnitsPerSystem) << endl << endl;
	
	
	// Global memory management
	SizeOfTimeDomain               = KernelParameters.NumberOfSystems * 2;
	SizeOfActualState              = KernelParameters.ThreadAllocationRequired * KernelParameters.UnitDimension;
	SizeOfUnitParameters           = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfUnitParameters;
	SizeOfSystemParameters         = KernelParameters.NumberOfSystems * KernelParameters.NumberOfSystemParameters;
	SizeOfGlobalParameters         = KernelParameters.NumberOfGlobalParameters;
	SizeOfIntegerGlobalParameters  = KernelParameters.NumberOfIntegerGlobalParameters;
	SizeOfUnitAccessories          = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfUnitAccessories;
	SizeOfIntegerUnitAccessories   = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfIntegerUnitAccessories;
	SizeOfSystemAccessories        = KernelParameters.NumberOfSystems * KernelParameters.NumberOfSystemAccessories;
	SizeOfIntegerSystemAccessories = KernelParameters.NumberOfSystems * KernelParameters.NumberOfIntegerSystemAccessories;
	SizeOfEvents                   = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfEvents;
	SizeOfCouplingMatrix           = KernelParameters.NumberOfCouplings * KernelParameters.UnitsPerSystem * KernelParameters.UnitsPerSystem;
	
	SizeOfDenseOutputIndex         = KernelParameters.NumberOfSystems;
	SizeOfDenseOutputTimeInstances = KernelParameters.NumberOfSystems * KernelParameters.DenseOutputNumberOfPoints;
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






#endif
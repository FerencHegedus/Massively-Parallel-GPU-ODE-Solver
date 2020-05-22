#ifndef SINGLESYSTEM_PERTHREAD_INTERFACE_H
#define SINGLESYSTEM_PERTHREAD_INTERFACE_H

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

enum Algorithms{ RK4, RKCK45 };

enum ListOfVariables{ All, \
                      TimeDomain, \
					  ActualState, \
					  ActualTime, \
					  ControlParameters, \
					  SharedParameters, \
					  IntegerSharedParameters, \
					  Accessories, \
					  IntegerAccessories, \
					  DenseOutput, \
					  DenseIndex, \
					  DenseTime, \
					  DenseState};

enum ListOfSolverOptions{ ThreadsPerBlock, \
						  InitialTimeStep, \
						  ActiveNumberOfThreads, \
                          MaximumTimeStep, \
						  MinimumTimeStep, \
						  TimeStepGrowLimit, \
						  TimeStepShrinkLimit, \
						  RelativeTolerance, \
						  AbsoluteTolerance, \
						  EventTolerance, \
						  EventDirection, \
						  DenseOutputMinimumTimeStep, \
						  DenseOutputSaveFrequency, \
						  PreferSharedMemory};

std::string SolverOptionsToString(ListOfSolverOptions);
std::string VariablesToString(ListOfVariables);

void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);

// Interface with the kernel
struct Struct_ThreadConfiguration
{
	int GridSize;
	int BlockSize;
	int NumberOfActiveThreads;
};

template <class Precision>
struct Struct_GlobalVariables
{
	Precision* d_TimeDomain;
	Precision* d_ActualState;
	Precision* d_ActualTime;
	Precision* d_ControlParameters;
	Precision* d_SharedParameters;
	int*       d_IntegerSharedParameters;
	Precision* d_Accessories;
	int*       d_IntegerAccessories;
	Precision* d_RelativeTolerance;
	Precision* d_AbsoluteTolerance;
	Precision* d_EventTolerance;
	int*       d_EventDirection;
	int*       d_DenseOutputIndex;
	Precision* d_DenseOutputTimeInstances;
	Precision* d_DenseOutputStates;
};

struct Struct_SharedMemoryUsage
{
	int PreferSharedMemory;  // Default: ON
	int IsAdaptive;
};

template <class Precision>
struct Struct_SolverOptions
{
	Precision InitialTimeStep;
	Precision MaximumTimeStep;
	Precision MinimumTimeStep;
	Precision TimeStepGrowLimit;
	Precision TimeStepShrinkLimit;
	int       DenseOutputSaveFrequency;
	Precision DenseOutputMinimumTimeStep;
};

template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
class ProblemSolver
{
    private:
		// Setup CUDA
		int Device;
		int Revision;
		cudaStream_t Stream;
		cudaEvent_t Event;
		
		// Thread management
		Struct_ThreadConfiguration ThreadConfiguration;
		
		// Global memory management
		size_t GlobalMemoryRequired;
		size_t GlobalMemoryFree;
		size_t GlobalMemoryTotal;
		
		long SizeOfTimeDomain;
		long SizeOfActualState;
		long SizeOfActualTime;
		long SizeOfControlParameters;
		long SizeOfSharedParameters;
		long SizeOfIntegerSharedParameters;
		long SizeOfAccessories;
		long SizeOfIntegerAccessories;
		long SizeOfEvents;
		long SizeOfDenseOutputIndex;
		long SizeOfDenseOutputTimeInstances;
		long SizeOfDenseOutputStates;
		
		Precision* h_TimeDomain;
		Precision* h_ActualState;
		Precision* h_ActualTime;
		Precision* h_ControlParameters;
		Precision* h_SharedParameters;
		int*       h_IntegerSharedParameters;
		Precision* h_Accessories;
		int*       h_IntegerAccessories;
		int*       h_DenseOutputIndex;
		Precision* h_DenseOutputTimeInstances;
		Precision* h_DenseOutputStates;
		
		Struct_GlobalVariables<Precision> GlobalVariables;
		
		// Shared memory management
		Struct_SharedMemoryUsage SharedMemoryUsage;
		
		size_t SharedMemoryRequiredSharedVariables;
		size_t SharedMemoryRequiredUpperLimit;
		size_t SharedMemoryRequired;
		size_t SharedMemoryAvailable;
		size_t DynamicSharedMemoryRequired;
		size_t StaticSharedMemoryRequired;
		
		// Default solver options
		Struct_SolverOptions<Precision> SolverOptions;
		
		// Private member functions
		void BoundCheck(std::string, std::string, int, int, int);
		void SharedMemoryCheck();
		
		template <typename T> void WriteToFileUniteScope(std::string, int, int, T*);
		template <typename T> void WriteToFileSystemAndGlobalScope(std::string, int, int, T*);
		template <typename T> void WriteToFileDenseOutput(std::string, int, int, T*, T*);
		
	public:
		ProblemSolver(int);
		~ProblemSolver();
		
		template <typename T> void SolverOption(ListOfSolverOptions, T);
		template <typename T> void SolverOption(ListOfSolverOptions, int, T);
		
		template <typename T> void SetHost(int, ListOfVariables, int, T);      // Problem scope and dense time
		template <typename T> void SetHost(ListOfVariables, int, T);           // Global scope
		template <typename T> void SetHost(int, ListOfVariables, int, int, T); // Dense state
		template <typename T> void SetHost(int, ListOfVariables, T);           // Dense index
		
		void SynchroniseFromHostToDevice(ListOfVariables);
		void SynchroniseFromDeviceToHost(ListOfVariables);
		
		template <typename T> T GetHost(int, ListOfVariables, int);            // Problem scope and dense time
		template <typename T> T GetHost(ListOfVariables, int);                 // Global scope
		template <typename T> T GetHost(int, ListOfVariables, int, int);       // Dense state
		template <typename T> T GetHost(int, ListOfVariables);                 // Dense index
		
		void Print(ListOfVariables);      // Problem and global scope
		void Print(ListOfVariables, int); // Dense output
		
		void Solve();
		
		void SynchroniseDevice();
		void InsertSynchronisationPoint();
		void SynchroniseSolver();
};


// --- INCLUDE SOLVERS ---

//#include "SingleSystem_PerThread_RungeKutta.cuh"


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
    std::cout << "---------------------------" << std::endl;
	std::cout << "Creating a SolverObject ..." << std::endl;
	std::cout << "---------------------------" << std::endl << std::endl;
	
	
	// ARCHITECTURE SPECIFIC SETUP
	std::cout << "ARCHITECTURE SPECIFIC SETUP:" << std::endl;
	
	Device = AssociatedDevice;
	gpuErrCHK( cudaSetDevice(Device) );
	gpuErrCHK( cudaStreamCreate(&Stream) );
	gpuErrCHK( cudaEventCreate(&Event) );
	
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, Device);
	
	Revision = SelectedDeviceProperties.major*10 + SelectedDeviceProperties.minor;
	std::cout << "   Compute capability of the selected device is: " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << std::endl;
	std::cout << "   It is advised to set a compiler option falg:  " << "--gpu-architecture=sm_" << Revision << std::endl << std::endl;
	
	if ( typeid(Precision) == typeid(double) )
		gpuErrCHK( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
	
	if ( typeid(Precision) == typeid(float) )
		gpuErrCHK( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );
	
	
	// THREAD MANAGEMENT
	std::cout << "THREAD MANAGEMENT:" << std::endl;
	
	ThreadConfiguration.NumberOfActiveThreads = NT;
	ThreadConfiguration.BlockSize             = SelectedDeviceProperties.warpSize; // Default option ThreadsPerBlock is the warp size
	ThreadConfiguration.GridSize              = NT/ThreadConfiguration.BlockSize + (NT % ThreadConfiguration.BlockSize == 0 ? 0:1);
	
	std::cout << "   Total number of threads:            " << std::setw(6) << NT << std::endl;
	std::cout << "   Active threads:                     " << std::setw(6) << NT                            << " (default: total number of threads) -> SolverOption: ActiveNumberOfThreads" << std::endl;
	std::cout << "   BlockSize (threads per block):      " << std::setw(6) << ThreadConfiguration.BlockSize << " (default: warp size)               -> SolverOption: ThreadsPerBlock" << std::endl;
	std::cout << "   GridSize (total number of blocks):  " << std::setw(6) << ThreadConfiguration.GridSize << std::endl << std::endl;
	
	
	// GLOBAL MEMORY MANAGEMENT
	std::cout << "GLOBAL MEMORY MANAGEMENT:" << std::endl;
	
	SizeOfTimeDomain               = NT * 2;
	SizeOfActualState              = NT * SD;
	SizeOfActualTime               = NT;
	SizeOfControlParameters        = NT * NCP;
	SizeOfSharedParameters         = NSP;
	SizeOfIntegerSharedParameters  = NT * NISP;
	SizeOfAccessories              = NT * NA;
	SizeOfIntegerAccessories       = NT * NIA;
	SizeOfEvents                   = NT * NE;
	SizeOfDenseOutputIndex         = NT;
	SizeOfDenseOutputTimeInstances = NT * NDO;
	SizeOfDenseOutputStates        = NT * SD * NDO;
	
	GlobalMemoryRequired = sizeof(Precision) * ( SizeOfTimeDomain + \
												 SizeOfActualState + \
												 SizeOfActualTime + \
												 SizeOfControlParameters + \
												 SizeOfSharedParameters + \
												 SizeOfAccessories + \
												 SizeOfDenseOutputTimeInstances + \
												 SizeOfDenseOutputStates + \
												 2*SD + NE) + \
						   sizeof(int) * ( SizeOfIntegerSharedParameters + \
										   SizeOfIntegerAccessories + \
										   SizeOfDenseOutputIndex + \
										   NE);
	
	cudaMemGetInfo( &GlobalMemoryFree, &GlobalMemoryTotal );
	std::cout << "   Required global memory:       " << GlobalMemoryRequired/1024/1024 << " Mb" << std::endl;
	std::cout << "   Available free global memory: " << GlobalMemoryFree/1024/1024     << " Mb" << std::endl;
	std::cout << "   Total global memory:          " << GlobalMemoryTotal/1024/1024    << " Mb" << std::endl;
	std::cout << "   Keep in mind that the real global memory usage can be higher according to the amount of register spilling!" << std::endl;
	
	if ( GlobalMemoryRequired >= GlobalMemoryFree )
	{
        std::cout << std::endl;
		std::cerr << "   ERROR: The required amount of global memory is larger than the free!" << std::endl;
		std::cerr << "          Try to reduce the number of points of the DenseOutput or reduce the NumberOfSystems!" << std::endl;
		std::cerr << "          Keep in mind that launch of multiple SolverObjects can also cause overuse of global memory!" << std::endl;
		std::cerr << "          (For instance, using 2 identically setup SolverObjects, the available global memory is halved!)" << std::endl;
        exit(EXIT_FAILURE);
    }
	std::cout << std::endl;
	
	h_TimeDomain               = AllocateHostPinnedMemory<Precision>( SizeOfTimeDomain );
	h_ActualState              = AllocateHostPinnedMemory<Precision>( SizeOfActualState );
	h_ActualTime               = AllocateHostPinnedMemory<Precision>( SizeOfActualTime );
	h_ControlParameters        = AllocateHostPinnedMemory<Precision>( SizeOfControlParameters );
	h_SharedParameters         = AllocateHostPinnedMemory<Precision>( SizeOfSharedParameters );
	h_IntegerSharedParameters  = AllocateHostPinnedMemory<int>( SizeOfIntegerSharedParameters );
	h_Accessories              = AllocateHostPinnedMemory<Precision>( SizeOfAccessories );
	h_IntegerAccessories       = AllocateHostPinnedMemory<int>( SizeOfIntegerAccessories );
	h_DenseOutputIndex         = AllocateHostPinnedMemory<int>( SizeOfDenseOutputIndex );
	h_DenseOutputTimeInstances = AllocateHostPinnedMemory<Precision>( SizeOfDenseOutputTimeInstances );
	h_DenseOutputStates        = AllocateHostPinnedMemory<Precision>( SizeOfDenseOutputStates );
	
	GlobalVariables.d_TimeDomain               = AllocateDeviceMemory<Precision>( SizeOfTimeDomain );
	GlobalVariables.d_ActualState              = AllocateDeviceMemory<Precision>( SizeOfActualState );
	GlobalVariables.d_ActualTime               = AllocateDeviceMemory<Precision>( SizeOfActualTime );
	GlobalVariables.d_ControlParameters        = AllocateDeviceMemory<Precision>( SizeOfControlParameters );
	GlobalVariables.d_SharedParameters         = AllocateDeviceMemory<Precision>( SizeOfSharedParameters );
	GlobalVariables.d_IntegerSharedParameters  = AllocateDeviceMemory<int>( SizeOfIntegerSharedParameters );
	GlobalVariables.d_Accessories              = AllocateDeviceMemory<Precision>( SizeOfAccessories );
	GlobalVariables.d_IntegerAccessories       = AllocateDeviceMemory<int>( SizeOfIntegerAccessories );
	GlobalVariables.d_RelativeTolerance        = AllocateDeviceMemory<Precision>( SD );
	GlobalVariables.d_AbsoluteTolerance        = AllocateDeviceMemory<Precision>( SD );
	GlobalVariables.d_EventTolerance           = AllocateDeviceMemory<Precision>( NE );
	GlobalVariables.d_EventDirection           = AllocateDeviceMemory<int>( NE );
	GlobalVariables.d_DenseOutputIndex         = AllocateDeviceMemory<int>( SizeOfDenseOutputIndex );
	GlobalVariables.d_DenseOutputTimeInstances = AllocateDeviceMemory<Precision>( SizeOfDenseOutputTimeInstances );
	GlobalVariables.d_DenseOutputStates        = AllocateDeviceMemory<Precision>( SizeOfDenseOutputStates );
	
	
	// SHARED MEMORY MANAGEMENT
	std::cout << "SHARED MEMORY MANAGEMENT:" << std::endl;
	
	switch (Algorithm)
	{
		case RK4:
			SharedMemoryUsage.IsAdaptive = 0;
			break;
		default:
			SharedMemoryUsage.IsAdaptive = 1;
			break;
	}
	
	SharedMemoryUsage.PreferSharedMemory = 1; // Default: ON
	SharedMemoryRequiredSharedVariables  = sizeof(Precision)*(NSP) + sizeof(int)*(NISP);
	DynamicSharedMemoryRequired          = SharedMemoryUsage.PreferSharedMemory * SharedMemoryRequiredSharedVariables;
	
	StaticSharedMemoryRequired           = sizeof(Precision)*( SharedMemoryUsage.IsAdaptive==0 ? 1 : SD ) + \	// AbsoluteTolerance
										   sizeof(Precision)*( SharedMemoryUsage.IsAdaptive==0 ? 1 : SD ) + \	// RelativeTolerance
										   sizeof(Precision)*( NE==0 ? 1 : NE ) + \								// EventTolerance
										   sizeof(int)*( NE==0 ? 1 : NE );										// EventDirection
	
	SharedMemoryRequiredUpperLimit       = StaticSharedMemoryRequired + SharedMemoryRequiredSharedVariables;
	SharedMemoryRequired                 = DynamicSharedMemoryRequired + StaticSharedMemoryRequired;
	SharedMemoryAvailable                = SelectedDeviceProperties.sharedMemPerBlock;
	
	std::cout << "   Required shared memory per block for managable variables:" << std::endl;
	std::cout << "    Shared memory required by shared variables:           " << std::setw(6) << SharedMemoryRequiredSharedVariables << " b (" << ( (SharedMemoryUsage.PreferSharedMemory == 0) ? "OFF" : "ON " ) << " -> SolverOption)" << std::endl;
	std::cout << "   Upper limit of possible shared memory usage per block: " << std::setw(6) << SharedMemoryRequiredUpperLimit << " b (Internals + PreferSharedMemory is ON)" << std::endl;
	std::cout << "   Actual shared memory required per block (estimated):   " << std::setw(6) << SharedMemoryRequired << " b" << std::endl;
	std::cout << "   Available shared memory per block:                     " << std::setw(6) << SharedMemoryAvailable << " b" << std::endl << std::endl;
	
	std::cout << "   Number of possible blocks per streaming multiprocessor: " << SharedMemoryAvailable/SharedMemoryRequired << std::endl;
	
	if ( SharedMemoryRequired >= SharedMemoryAvailable )
	{
        std::cout << std::endl;
		std::cout << "   WARNING: The required amount of shared memory is larger than the available!" << std::endl;
		std::cout << "            The solver kernel function cannot be run on the selected GPU!" << std::endl;
		std::cout << "            Turn OFF some variables using shared memory!" << std::endl;
    }
	std::cout << std::endl;
	
	
	// DEFAULT VALUES OF SOLVER OPTIONS
	std::cout << "DEFAULT SOLVER OPTIONS:" << std::endl;
	
	SolverOptions.InitialTimeStep            = 1e-2;
	SolverOptions.MaximumTimeStep            = 1.0e6;
	SolverOptions.MinimumTimeStep            = 1.0e-12;
	SolverOptions.TimeStepGrowLimit          = 5.0;
	SolverOptions.TimeStepShrinkLimit        = 0.1;
	SolverOptions.DenseOutputMinimumTimeStep = 0.0;
	SolverOptions.DenseOutputSaveFrequency   = 1;
	
	Precision DefaultAlgorithmTolerances = 1e-8;
	for (int i=0; i<SD; i++)
	{
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_RelativeTolerance + i, &DefaultAlgorithmTolerances, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_AbsoluteTolerance + i, &DefaultAlgorithmTolerances, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
	}
	
	Precision DefaultEventTolerance = 1e-6;
	int       DefaultEventDirection = 0;
	for (int i=0; i<NE; i++)
	{
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventTolerance   + i, &DefaultEventTolerance, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventDirection   + i, &DefaultEventDirection, sizeof(int),       cudaMemcpyHostToDevice, Stream) );
	}
	
	std::cout << "   Active threads:                           " << std::setw(6) << NT << " (default: total number of threads) -> SolverOption: ActiveNumberOfThreads" << std::endl;
	std::cout << "   BlockSize (threads per block):            " << std::setw(6) << ThreadConfiguration.BlockSize << " (default: warp size)               -> SolverOption: ThreadsPerBlock" << std::endl;
	std::cout << "   Initial time step:                        " << std::setw(6) << SolverOptions.InitialTimeStep << std::endl;
	std::cout << "   Maximum time step:                        " << std::setw(6) << SolverOptions.MaximumTimeStep << std::endl;
	std::cout << "   Minimum time step:                        " << std::setw(6) << SolverOptions.MinimumTimeStep << std::endl;
	std::cout << "   Time step grow limit:                     " << std::setw(6) << SolverOptions.TimeStepGrowLimit << std::endl;
	std::cout << "   Time step shrink limit:                   " << std::setw(6) << SolverOptions.TimeStepShrinkLimit << std::endl;
	std::cout << "   Dense output minimum time step:           " << std::setw(6) << SolverOptions.DenseOutputMinimumTimeStep << std::endl;
	std::cout << "   Dense output save frequency:              " << std::setw(6) << SolverOptions.DenseOutputSaveFrequency << std::endl;
	std::cout << "   Algorithm absolute tolerance (all comp.): " << std::setw(6) << 1e-8 << std::endl;
	std::cout << "   Algorithm relative tolerance (all comp.): " << std::setw(6) << 1e-8 << std::endl;
	std::cout << "   Event absolute tolerance:                 " << std::setw(6) << 1e-6 << std::endl;
	std::cout << "   Event direction of detection:             " << std::setw(6) << 0 << std::endl;
	
	std::cout << std::endl;
	std::cout << "---------------------------------------------------" << std::endl;
	std::cout << "Object for Parameters scan is successfully created!" << std::endl;
	std::cout << "Required memory allocations have been done" << std::endl;
	std::cout << "Coo man coo!!!" << std::endl;
	std::cout << "---------------------------------------------------" << std::endl << std::endl;
}

// DESTRUCTOR
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::~ProblemSolver()
{
    /*gpuErrCHK( cudaSetDevice(Device) );
	
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
	gpuErrCHK( cudaFree(KernelParameters.d_DenseOutputStates) );*/
	
	std::cout << "Object for Parameters scan is deleted! Every memory have been deallocated!" << std::endl << std::endl;
}
/*
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber, double Value)
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber, int TimeStep, double Value)
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(ListOfVariables Variable, int SerialNumber, double Value)
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromHostToDevice(ListOfVariables Variable)
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromDeviceToHost(ListOfVariables Variable)
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
double ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber)
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
double ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber, int TimeStep)
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
double ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(ListOfVariables Variable, int SerialNumber)
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(ListOfVariables Variable)
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
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(ListOfVariables Variable, int ThreadID)
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
	
	//SingleSystem_PerThread_RungeKutta<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision><<<GridSize, BlockSize, DynamicSharedMemory, Stream>>> (KernelParameters);
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
*/
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

std::string SolverOptionsToString(ListOfSolverOptions Option)
{
	switch(Option)
	{
		case ThreadsPerBlock:
			return "ThreadsPerBlock";
		case InitialTimeStep:
			return "InitialTimeStep";
		case ActiveNumberOfThreads:
			return "ActiveNumberOfThreads";
		case MaximumTimeStep:
			return "MaximumTimeStep";
		case MinimumTimeStep:
			return "MinimumTimeStep";
		case TimeStepGrowLimit:
			return "TimeStepGrowLimit";
		case TimeStepShrinkLimit:
			return "TimeStepShrinkLimit";
		case RelativeTolerance:
			return "RelativeTolerance";
		case AbsoluteTolerance:
			return "AbsoluteTolerance";
		case EventTolerance:
			return "EventTolerance";
		case EventDirection:
			return "EventDirection";
		case DenseOutputMinimumTimeStep:
			return "DenseOutputMinimumTimeStep";
		case DenseOutputSaveFrequency:
			return "DenseOutputSaveFrequency";
		case PreferSharedMemory:
			return "PreferSharedMemory";
		default:
			return "Non-existent solver option!";
	}
}

std::string VariablesToString(ListOfVariables Option)
{
	switch(Option)
	{
		case All:
			return "All";
		case TimeDomain:
			return "TimeDomain";
		case ActualState:
			return "ActualState";
		case ActualTime:
			return "ActualTime";
		case ControlParameters:
			return "ControlParameters";
		case SharedParameters:
			return "SharedParameters";
		case IntegerSharedParameters:
			return "IntegerSharedParameters";
		case Accessories:
			return "Accessories";
		case IntegerAccessories:
			return "IntegerAccessories";
		case DenseOutput:
			return "DenseOutput";
		case DenseIndex:
			return "DenseIndex";
		case DenseTime:
			return "DenseTime";
		case DenseState:
			return "DenseState";
		default:
			return "Non-existent variable!";
	}
}

#endif
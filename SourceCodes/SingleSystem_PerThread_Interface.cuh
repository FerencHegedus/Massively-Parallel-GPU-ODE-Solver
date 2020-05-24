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
		
		// Auxiliary member functions
		void BoundCheck(std::string, std::string, int, int, int);
		void SharedMemoryCheck();
		template <typename T> void WriteToFileGeneral(std::string, int, int, T*);
		
	public:
		ProblemSolver(int);
		~ProblemSolver();
		
		template <typename T> void SolverOption(ListOfSolverOptions, T);
		template <typename T> void SolverOption(ListOfSolverOptions, int, T);
		
		template <typename T> void SetHost(int, ListOfVariables, int, T);      // Unit scope and DenseTime
		template <typename T> void SetHost(int, ListOfVariables, T);           // System scope
		template <typename T> void SetHost(ListOfVariables, int, T);           // Global scope
		template <typename T> void SetHost(int, ListOfVariables, int, int, T); // DenseState
		
		void SynchroniseFromHostToDevice(ListOfVariables);
		void SynchroniseFromDeviceToHost(ListOfVariables);
		
		template <typename T> T GetHost(int, ListOfVariables, int);            // Unit scope and DenseTime
		template <typename T> T GetHost(int, ListOfVariables);                 // System scope
		template <typename T> T GetHost(ListOfVariables, int);                 // Global scope
		template <typename T> T GetHost(int, ListOfVariables, int, int);       // DenseState
		
		void Print(ListOfVariables);      // General
		void Print(ListOfVariables, int); // DenseOutput
		
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
		std::cout << "            Turn OFF PreferSharedMemory!" << std::endl;
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
    gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaStreamDestroy(Stream) );
	gpuErrCHK( cudaEventDestroy(Event) );
	
	gpuErrCHK( cudaFreeHost(h_TimeDomain) );
	gpuErrCHK( cudaFreeHost(h_ActualState) );
	gpuErrCHK( cudaFreeHost(h_ActualTime) );
	gpuErrCHK( cudaFreeHost(h_ControlParameters) );
	gpuErrCHK( cudaFreeHost(h_SharedParameters) );
	gpuErrCHK( cudaFreeHost(h_IntegerSharedParameters) );
	gpuErrCHK( cudaFreeHost(h_Accessories) );
	gpuErrCHK( cudaFreeHost(h_IntegerAccessories) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputIndex) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputStates) );
	
	gpuErrCHK( cudaFree(GlobalVariables.d_TimeDomain) );
	gpuErrCHK( cudaFree(GlobalVariables.d_ActualState) );
	gpuErrCHK( cudaFree(GlobalVariables.d_ActualTime) );
	gpuErrCHK( cudaFree(GlobalVariables.d_ControlParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_SharedParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerSharedParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_Accessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_RelativeTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_AbsoluteTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventDirection) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputIndex) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputStates) );
	
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Object for Parameters scan is deleted!" << std::endl;
	std::cout << "Every memory have been deallocated!" << std::endl;
	std::cout << "Coo man coo!!!" << std::endl;
	std::cout << "--------------------------------------" << std::endl << std::endl;
}

// BOUND CHECK, set/get host, options
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::BoundCheck(std::string Function, std::string Variable, int Value, int LowerLimit, int UpperLimit)
{
	if ( ( Value < LowerLimit ) || ( Value > UpperLimit ) )
	{
        std::cerr << "ERROR: In solver member function " << Function << "!" << std::endl;
		std::cerr << "       Option: " << Variable << std::endl;
		
		if ( LowerLimit>UpperLimit )
			std::cerr << "       Acceptable index: none" << std::endl;
		else
			std::cerr << "       Acceptable index: " << LowerLimit << "-" << UpperLimit << std::endl;
		
		std::cerr << "       Current index:    " << Value << std::endl;
        exit(EXIT_FAILURE);
    }
}

// SHARED MEMORY CHECK
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SharedMemoryCheck()
{
	std::cout << "SHARED MEMORY MANAGEMENT CHANGED BY SOLVER OPTION:" << std::endl;
	
	DynamicSharedMemoryRequired = SharedMemoryUsage.PreferSharedMemory * SharedMemoryRequiredSharedVariables;
	SharedMemoryRequired        = DynamicSharedMemoryRequired + StaticSharedMemoryRequired;
	
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
		std::cout << "            Turn OFF PreferSharedMemory!" << std::endl;
    }
	std::cout << std::endl;
}

// OPTION, single input argument
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, T Value)
{
	switch (Option)
	{
		case ThreadsPerBlock:
			ThreadConfiguration.BlockSize = (int)Value;
			ThreadConfiguration.GridSize  = NT/ThreadConfiguration.BlockSize + (NT % ThreadConfiguration.BlockSize == 0 ? 0:1);
			break;
		
		case InitialTimeStep:
			SolverOptions.InitialTimeStep = (int)Value;
			break;
		
		case ActiveNumberOfThreads:
			ThreadConfiguration.NumberOfActiveThreads = (int)Value;
			break;
		
		case MaximumTimeStep:
			SolverOptions.MaximumTimeStep = (double)Value;
			break;
		
		case MinimumTimeStep:
			SolverOptions.MinimumTimeStep = (double)Value;
			break;
		
		case TimeStepGrowLimit:
			SolverOptions.TimeStepGrowLimit = (double)Value;
			break;
		
		case TimeStepShrinkLimit:
			SolverOptions.TimeStepShrinkLimit = (double)Value;
			break;
		
		case DenseOutputMinimumTimeStep:
			SolverOptions.DenseOutputMinimumTimeStep = (double)Value;
			break;
		
		case DenseOutputSaveFrequency:
			SolverOptions.DenseOutputSaveFrequency = (int)Value;
			break;
		
		case PreferSharedMemory:
			SharedMemoryUsage.PreferSharedMemory = (int)Value;
			SharedMemoryCheck();
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SolverOption!" << std::endl;
			std::cerr << "       Option: " << SolverOptionsToString(Option) << std::endl;
			std::cerr << "       This option needs 2 input arguments or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// OPTION, double input argument
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, int Index, T Value)
{
	Precision PValue = (Precision)Value;
	int       IValue = (int)Value;
	
	switch (Option)
	{
		case RelativeTolerance:
			BoundCheck("SolverOption", "RelativeTolerance", Index, 0, SD-1);
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_RelativeTolerance+Index, &PValue, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case AbsoluteTolerance:
			BoundCheck("SolverOption", "AbsoluteTolerance", Index, 0, SD-1);
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_AbsoluteTolerance+Index, &PValue, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case EventTolerance:
			BoundCheck("SolverOption", "EventTolerance", Index, 0, NE-1);
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventTolerance+Index, &PValue, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case EventDirection:
			BoundCheck("SolverOption", "EventDirection", Index, 0, NE-1);
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventDirection+Index, &IValue, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SolverOption!" << std::endl;
			std::cerr << "       Option: " << SolverOptionsToString(Option) << std::endl;
			std::cerr << "       This option needs 1 input arguments or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Unit scope and DenseTime
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber, T Value)
{
	BoundCheck("SetHost", "ProblemNumber", ProblemNumber, 0, NT-1 );
	
	int idx = ProblemNumber + SerialNumber*NT;
	
	switch (Variable)
	{
		case TimeDomain:
			BoundCheck("SetHost", "TimeDomain", SerialNumber, 0, 1 );
			h_TimeDomain[idx] = (Precision)Value;
			break;
		
		case ActualState:
			BoundCheck("SetHost", "ActualState", SerialNumber, 0, SD-1 );
			h_ActualState[idx] = (Precision)Value;
			break;
		
		case ControlParameters:
			BoundCheck("SetHost", "ControlParameters", SerialNumber, 0, NCP-1 );
			h_ControlParameters[idx] = (Precision)Value;
			break;
		
		case Accessories:
			BoundCheck("SetHost", "Accessories", SerialNumber, 0, NA-1 );
			h_Accessories[idx] = (Precision)Value;
			break;
		
		case IntegerAccessories:
			BoundCheck("SetHost", "IntegerAccessories", SerialNumber, 0, NIA-1 );
			h_IntegerAccessories[idx] = (int)Value;
			break;
		
		case DenseTime:
			BoundCheck("SetHost", "DenseTime", SerialNumber, 0, NDO-1 );
			h_DenseOutputTimeInstances[idx] = (Precision)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, System scope
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, ListOfVariables Variable, T Value)
{
	BoundCheck("SetHost", "ProblemNumber", ProblemNumber, 0, NT-1 );
	
	int idx = ProblemNumber;
	
	switch (Variable)
	{
		case ActualTime:
			h_ActualTime[idx] = (Precision)Value;
			break;
		
		case DenseIndex:
			h_DenseOutputIndex[idx] = (int)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Global scope
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(ListOfVariables Variable, int SerialNumber, T Value)
{
	switch (Variable)
	{
		case SharedParameters:
			BoundCheck("SetHost", "SharedParameters", SerialNumber, 0, NSP-1 );
			h_SharedParameters[SerialNumber] = (Precision)Value;
			break;
		
		case IntegerSharedParameters:
			BoundCheck("SetHost", "IntegerSharedParameters", SerialNumber, 0, NISP-1 );
			h_IntegerSharedParameters[SerialNumber] = (int)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, DenseState
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber, int TimeStepNumber, T Value)
{
	BoundCheck("SetHost", "ProblemNumber", ProblemNumber, 0, NT-1 );
	
	int idx = ProblemNumber + SerialNumber*NT + TimeStepNumber*NT*SD;
	
	switch (Variable)
	{
		case DenseState:
			BoundCheck("SetHost", "DenseState/ComponentNumber", SerialNumber, 0, SD-1 );
			BoundCheck("SetHost", "DenseState/TimeStepNumber", TimeStepNumber, 0, NDO-1 );
			h_DenseOutputStates[idx] = (Precision)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, Host -> Device
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromHostToDevice(ListOfVariables Variable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (Variable)
	{
		case TimeDomain:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_TimeDomain, h_TimeDomain, SizeOfTimeDomain*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case ActualState:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_ActualState, h_ActualState, SizeOfActualState*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case ActualTime:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_ActualTime, h_ActualTime, SizeOfActualTime*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case ControlParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_ControlParameters, h_ControlParameters, SizeOfControlParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case SharedParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_SharedParameters, h_SharedParameters, SizeOfSharedParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case IntegerSharedParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerSharedParameters, h_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case Accessories:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_Accessories, h_Accessories, SizeOfAccessories*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case IntegerAccessories:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerAccessories, h_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case DenseOutput:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_DenseOutputIndex, h_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_DenseOutputTimeInstances, h_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_DenseOutputStates, h_DenseOutputStates, SizeOfDenseOutputStates*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case All:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_TimeDomain, h_TimeDomain, SizeOfTimeDomain*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_ActualState, h_ActualState, SizeOfActualState*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_ActualTime, h_ActualTime, SizeOfActualTime*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_ControlParameters, h_ControlParameters, SizeOfControlParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_SharedParameters, h_SharedParameters, SizeOfSharedParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_Accessories, h_Accessories, SizeOfAccessories*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerSharedParameters, h_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerAccessories, h_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_DenseOutputIndex, h_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_DenseOutputTimeInstances, h_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_DenseOutputStates, h_DenseOutputStates, SizeOfDenseOutputStates*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SynchroniseFromHostToDevice!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option is not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, Device -> Host
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::SynchroniseFromDeviceToHost(ListOfVariables Variable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (Variable)
	{
		case TimeDomain:
			gpuErrCHK( cudaMemcpyAsync(h_TimeDomain, GlobalVariables.d_TimeDomain, SizeOfTimeDomain*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case ActualState:
			gpuErrCHK( cudaMemcpyAsync(h_ActualState, GlobalVariables.d_ActualState, SizeOfActualState*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case ActualTime:
			gpuErrCHK( cudaMemcpyAsync(h_ActualTime, GlobalVariables.d_ActualTime, SizeOfActualTime*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case ControlParameters:
			gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, GlobalVariables.d_ControlParameters, SizeOfControlParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case SharedParameters:
			gpuErrCHK( cudaMemcpyAsync(h_SharedParameters, GlobalVariables.d_SharedParameters, SizeOfSharedParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case IntegerSharedParameters:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerSharedParameters, GlobalVariables.d_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case Accessories:
			gpuErrCHK( cudaMemcpyAsync(h_Accessories, GlobalVariables.d_Accessories, SizeOfAccessories*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case IntegerAccessories:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerAccessories, GlobalVariables.d_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case DenseOutput:
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex, GlobalVariables.d_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, GlobalVariables.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates, GlobalVariables.d_DenseOutputStates, SizeOfDenseOutputStates*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case All:
			gpuErrCHK( cudaMemcpyAsync(h_TimeDomain, GlobalVariables.d_TimeDomain, SizeOfTimeDomain*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ActualState, GlobalVariables.d_ActualState, SizeOfActualState*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ActualTime, GlobalVariables.d_ActualTime, SizeOfActualTime*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, GlobalVariables.d_ControlParameters, SizeOfControlParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_SharedParameters, GlobalVariables.d_SharedParameters, SizeOfSharedParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_Accessories, GlobalVariables.d_Accessories, SizeOfAccessories*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerSharedParameters, GlobalVariables.d_IntegerSharedParameters, SizeOfIntegerSharedParameters*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerAccessories, GlobalVariables.d_IntegerAccessories, SizeOfIntegerAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex, GlobalVariables.d_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, GlobalVariables.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates, GlobalVariables.d_DenseOutputStates, SizeOfDenseOutputStates*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		default:
			std::cerr << "ERROR in solver member function SynchroniseFromDeviceToHost:" << std::endl << "    "\
			          << "Invalid option for variable selection!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Unit scope and DenseTime
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber)
{
	BoundCheck("SetHost", "ProblemNumber", ProblemNumber, 0, NT-1 );
	
	int idx = ProblemNumber + SerialNumber*NT;
	
	switch (Variable)
	{
		case TimeDomain:
			BoundCheck("SetHost", "TimeDomain", SerialNumber, 0, 1 );
			return (T)h_TimeDomain[idx];
		
		case ActualState:
			BoundCheck("SetHost", "ActualState", SerialNumber, 0, SD-1 );
			return (T)h_ActualState[idx];
		
		case ControlParameters:
			BoundCheck("SetHost", "ControlParameters", SerialNumber, 0, NCP-1 );
			return (T)h_ControlParameters[idx];
		
		case Accessories:
			BoundCheck("SetHost", "Accessories", SerialNumber, 0, NA-1 );
			return (T)h_Accessories[idx];
		
		case IntegerAccessories:
			BoundCheck("SetHost", "IntegerAccessories", SerialNumber, 0, NIA-1 );
			return (T)h_IntegerAccessories[idx];
		
		case DenseTime:
			BoundCheck("SetHost", "DenseTime", SerialNumber, 0, NDO-1 );
			return (T)h_DenseOutputTimeInstances[idx];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, System scope
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, ListOfVariables Variable)
{
	BoundCheck("SetHost", "ProblemNumber", ProblemNumber, 0, NT-1 );
	
	int idx = ProblemNumber;
	
	switch (Variable)
	{
		case ActualTime:
			return (T)h_ActualTime[idx];
		
		case DenseIndex:
			return (T)h_DenseOutputIndex[idx];
		
		default:
				std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
				std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
				std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
				exit(EXIT_FAILURE);
	}
}

// GETHOST, Global scope
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(ListOfVariables Variable, int SerialNumber)
{
	switch (Variable)
	{
		case SharedParameters:
			BoundCheck("SetHost", "SharedParameters", SerialNumber, 0, NSP-1 );
			return (T)h_SharedParameters[SerialNumber];
		
		case IntegerSharedParameters:
			BoundCheck("SetHost", "IntegerSharedParameters", SerialNumber, 0, NISP-1 );
			return (T)h_IntegerSharedParameters[SerialNumber];
		
		default:
				std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
				std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
				std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
				exit(EXIT_FAILURE);
	}
}

// GETHOST, DenseState
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::GetHost(int ProblemNumber, ListOfVariables Variable, int SerialNumber, int TimeStepNumber)
{
	BoundCheck("SetHost", "ProblemNumber", ProblemNumber, 0, NT-1 );
	
	int idx = ProblemNumber + SerialNumber*NT + TimeStepNumber*NT*SD;
	
	switch (Variable)
	{
		case DenseState:
			BoundCheck("SetHost", "DenseState/ComponentNumber", SerialNumber, 0, SD-1 );
			BoundCheck("SetHost", "DenseState/TimeStepNumber", TimeStepNumber, 0, NDO-1 );
			return (T)h_DenseOutputStates[idx];
		
		default:
				std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
				std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
				std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
				exit(EXIT_FAILURE);
	}
}

// WRITE TO FILE, General
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::WriteToFileGeneral(std::string FileName, int NumberOfRows, int NumberOfColumns, T* Data)
{
	std::ofstream DataFile;
	DataFile.open (FileName);
	
	// Make it depend on type
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int idx;
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			idx = i + j*NumberOfRows;
			DataFile.width(Width); DataFile << Data[idx];
			if ( j<(NumberOfColumns-1) )
				DataFile << ',';
		}
		DataFile << '\n';
	}
	
	DataFile.close();
}

// PRINT, General
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(ListOfVariables Variable)
{
	std::string FileName;
	int NumberOfRows;
	int NumberOfColumns;
	
	switch (Variable)
	{
		case TimeDomain:
			FileName = "TimeDomainInSolverObject.txt";
			NumberOfRows = NT;
			NumberOfColumns = 2;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_TimeDomain);
			break;
		
		case ActualState:
			FileName = "ActualStateInSolverObject.txt";
			NumberOfRows = NT;
			NumberOfColumns = SD;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_ActualState);
			break;
		
		case ActualTime:
			FileName = "ActualTimeInSolverObject.txt";
			NumberOfRows = NT;
			NumberOfColumns = 1;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_ActualTime);
			break;
		
		case ControlParameters:
			FileName = "ControlParametersInSolverObject.txt";
			NumberOfRows = NT;
			NumberOfColumns = NCP;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_ControlParameters);
			break;
		
		case SharedParameters:
			FileName = "SharedParametersInSolverObject.txt";
			NumberOfRows = NSP;		
			NumberOfColumns = 1;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_SharedParameters);
			break;
		
		case IntegerSharedParameters:
			FileName = "IntegerSharedParametersInSolverObject.txt";
			NumberOfRows = NISP;		
			NumberOfColumns = 1;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_IntegerSharedParameters);
			break;
		
		case Accessories:
			FileName = "AccessoriesInSolverObject.txt";
			NumberOfRows = NT;
			NumberOfColumns = NA;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_Accessories);
			break;
		
		case IntegerAccessories:
			FileName = "IntegerAccessoriesInSolverObject.txt";
			NumberOfRows = NT;
			NumberOfColumns = NIA;
			WriteToFileGeneral(FileName, NumberOfRows, NumberOfColumns, h_IntegerAccessories);
			break;
		
		default :
			std::cerr << "ERROR: In solver member function Print!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// PRINT, DenseOutput
template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,Algorithm,Precision>::Print(ListOfVariables Variable, int ThreadID)
{
	BoundCheck("Print", "Thread", ThreadID, 0, NT-1 );
	
	if ( Variable != DenseOutput )
	{
        std::cerr << "ERROR: In solver member function Print!" << std::endl;
		std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
		std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
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
	for (int i=0; i<NCP; i++)
	{
		idx = ThreadID + i*NT;
		DataFile.width(Width); DataFile << h_ControlParameters[idx];
		if ( i<(NCP-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "SharedParameters:\n";
	for (int i=0; i<NSP; i++)
	{
		DataFile.width(Width); DataFile << h_SharedParameters[i];
		if ( i<(NSP-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "IntegerSharedParameters:\n";
	for (int i=0; i<NISP; i++)
	{
		DataFile.width(Width); DataFile << h_IntegerSharedParameters[i];
		if ( i<(NISP-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "Accessories:\n";
	for (int i=0; i<NA; i++)
	{
		idx = ThreadID + i*NT;
		DataFile.width(Width); DataFile << h_Accessories[idx];
		if ( i<(NA-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	DataFile << "IntegerAccessories:\n";
	for (int i=0; i<NIA; i++)
	{
		idx = ThreadID + i*NT;
		DataFile.width(Width); DataFile << h_IntegerAccessories[idx];
		if ( i<(NIA-1) )
			DataFile << ',';
	}
	DataFile << "\n\n";
	
	DataFile << "Time series:\n";
	if ( NDO > 0 )
	{
		for (int i=0; i<(h_DenseOutputIndex[ThreadID]+1); i++)
		{
			idx = ThreadID + i*NT;
			DataFile.width(Width); DataFile << h_DenseOutputTimeInstances[idx] << ',';
			
			for (int j=0; j<SD; j++)
			{
				idx = ThreadID + j*NT + i*NT*SD;
				DataFile.width(Width); DataFile << h_DenseOutputStates[idx];
				if ( j<(SD-1) )
					DataFile << ',';
			}
			DataFile << '\n';
		}
	}
	
	DataFile.close();
}

// SOLVE
/*template <int NT, int SD, int NCP, int NSP, int NISP, int NE, int NA, int NIA, int NDO, Algorithms Algorithm, class Precision>
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
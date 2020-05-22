#ifndef COUPLEDSYSTEMS_PERBLOCK_INTERFACE_H
#define COUPLEDSYSTEMS_PERBLOCK_INTERFACE_H

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
					  ActualTime, \
					  ActualState, \
                      UnitParameters, \
					  SystemParameters, \
					  GlobalParameters, \
					  IntegerGlobalParameters, \
					  UnitAccessories, \
					  IntegerUnitAccessories, \
					  SystemAccessories, \
					  IntegerSystemAccessories, \
					  CouplingMatrix, \
					  CouplingStrength, \
					  CouplingIndex, \
					  DenseOutput, \
					  DenseIndex, \
					  DenseTime, \
					  DenseState};

enum ListOfSolverOptions{ InitialTimeStep, \
						  ActiveSystems, \
						  MaximumTimeStep, \
						  MinimumTimeStep, \
                          TimeStepGrowLimit, \
						  TimeStepShrinkLimit, \
						  RelativeTolerance, \
						  AbsoluteTolerance, \
						  EventTolerance, \
						  EventDirection, \
						  DenseOutputSaveFrequency, \
						  DenseOutputMinimumTimeStep, \
						  SharedGlobalVariables, \
						  SharedCouplingMatrices};
			
std::string SolverOptionsToString(ListOfSolverOptions);
std::string VariablesToString(ListOfVariables);

void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);

// Interface with the kernels
struct Struct_ThreadConfiguration
{
	int LogicalThreadsPerBlock;
	int NumberOfBlockLaunches;
	int ThreadPaddingPerBlock;
	int BlockSize;
	int GridSize;
	int TotalLogicalThreads;
};

template <class Precision>
struct Struct_GlobalVariables
{
	Precision* d_TimeDomain;
	Precision* d_ActualState;
	Precision* d_ActualTime;
	Precision* d_UnitParameters;
	Precision* d_SystemParameters;
	Precision* d_GlobalParameters;
	int*       d_IntegerGlobalParameters;
	Precision* d_UnitAccessories;
	int*       d_IntegerUnitAccessories;
	Precision* d_SystemAccessories;
	int*       d_IntegerSystemAccessories;
	Precision* d_CouplingMatrix;
	Precision* d_CouplingStrength;
	int*       d_CouplingIndex;
	int*       d_DenseOutputIndex;
	Precision* d_DenseOutputTimeInstances;
	Precision* d_DenseOutputStates;
	Precision* d_RelativeTolerance;
	Precision* d_AbsoluteTolerance;
	Precision* d_EventTolerance;
	int*       d_EventDirection;
};

struct Struct_SharedMemoryUsage
{
	bool GlobalVariables;  // Default: ON
	bool CouplingMatrices; // Default: OFF
	int  SingleCouplingMatrixSize;
	int  IsAdaptive;
};

template <class Precision>
struct Struct_SolverOptions
{
	Precision InitialTimeStep;
	int       ActiveSystems;
	Precision MaximumTimeStep;
	Precision MinimumTimeStep;
	Precision TimeStepGrowLimit;
	Precision TimeStepShrinkLimit;
	int       DenseOutputSaveFrequency;
	Precision DenseOutputMinimumTimeStep;
};

template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
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
		long SizeOfUnitParameters;
		long SizeOfSystemParameters;
		long SizeOfGlobalParameters;
		long SizeOfIntegerGlobalParameters;
		long SizeOfUnitAccessories;
		long SizeOfIntegerUnitAccessories;
		long SizeOfSystemAccessories;
		long SizeOfIntegerSystemAccessories;
		long SizeOfEvents;
		long SizeOfCouplingMatrix;
		long SizeOfCouplingMatrixOriginal;
		long SizeOfCouplingStrength;
		long SizeOfCouplingIndex;
		long SizeOfDenseOutputIndex;
		long SizeOfDenseOutputTimeInstances;
		long SizeOfDenseOutputStates;
		
		Precision* h_TimeDomain;               // System scope
		Precision* h_ActualState;              // Unit scope
		Precision* h_ActualTime;               // System scope
		Precision* h_UnitParameters;           // Unit scope
		Precision* h_SystemParameters;         // System scope
		Precision* h_GlobalParameters;         // Global scope
		int*       h_IntegerGlobalParameters;  // Global scope
		Precision* h_UnitAccessories;          // Unit scope
		int*       h_IntegerUnitAccessories;   // Unit scope
		Precision* h_SystemAccessories;        // System scope
		int*       h_IntegerSystemAccessories; // System scope
		Precision* h_CouplingMatrix;           // Global scope
		Precision* h_CouplingMatrixOriginal;   // ONLY IN HOST
		Precision* h_CouplingStrength;         // System scope
		int*       h_CouplingIndex;            // Global scope
		int*       h_DenseOutputIndex;         // System scope
		Precision* h_DenseOutputTimeInstances; // System scope
		Precision* h_DenseOutputStates;        // Unit scope
		
		Struct_GlobalVariables<Precision> GlobalVariables;
		
		// Shared memory management
		Struct_SharedMemoryUsage SharedMemoryUsage;
		
		size_t SharedMemoryRequiredGlobalVariables;
		size_t SharedMemoryRequiredCouplingMatrices;
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
		
		template <typename T> void SetHost(int, int, ListOfVariables, int, T);      // Unit scope
		template <typename T> void SetHost(int, ListOfVariables, int, T);           // System scope and dense time
		template <typename T> void SetHost(ListOfVariables, int, T);                // Global scope
		template <typename T> void SetHost(int, ListOfVariables, int, int, T);      // Coupling matrix
		template <typename T> void SetHost(int, ListOfVariables, T);                // Dense index
		template <typename T> void SetHost(int, int, ListOfVariables, int, int, T); // Dense state
		
		void SynchroniseFromHostToDevice(ListOfVariables);
		void SynchroniseFromDeviceToHost(ListOfVariables);
		
		template <typename T> T GetHost(int, int, ListOfVariables, int);      // Unit scope
		template <typename T> T GetHost(int, ListOfVariables, int);           // System scope and dense time
		template <typename T> T GetHost(ListOfVariables, int);                // Global scope
		template <typename T> T GetHost(int, ListOfVariables, int, int);      // Coupling matrix
		template <typename T> T GetHost(int, ListOfVariables);                // Dense index
		template <typename T> T GetHost(int, int, ListOfVariables, int, int); // Dense state
		
		void Print(ListOfVariables);      // Unit, system and global scope
		void Print(ListOfVariables, int); // Coupling matrix, dense output
		
		void Solve();
		
		void SynchroniseDevice();
		void InsertSynchronisationPoint();
		void SynchroniseSolver();
};


// --- INCLUDE SOLVERS ---


#include "CoupledSystmes_PerBlock_Solver.cuh"


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
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::ProblemSolver(int AssociatedDevice)
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
	
	ThreadConfiguration.LogicalThreadsPerBlock = SPB * UPS;
	ThreadConfiguration.NumberOfBlockLaunches  = ThreadConfiguration.LogicalThreadsPerBlock / TPB + (ThreadConfiguration.LogicalThreadsPerBlock % TPB == 0 ? 0:1);
	ThreadConfiguration.ThreadPaddingPerBlock  = ThreadConfiguration.NumberOfBlockLaunches * TPB - ThreadConfiguration.LogicalThreadsPerBlock;
	ThreadConfiguration.BlockSize              = TPB;
	ThreadConfiguration.GridSize               = NS/SPB + (NS % SPB == 0 ? 0 : 1);
	ThreadConfiguration.TotalLogicalThreads    = (ThreadConfiguration.LogicalThreadsPerBlock + ThreadConfiguration.ThreadPaddingPerBlock) * ThreadConfiguration.GridSize;
	
	std::cout << "   Total number of systems:            " << std::setw(6) << NS << std::endl;
	std::cout << "   Systems per block:                  " << std::setw(6) << SPB << std::endl;
	std::cout << "   Logical threads per block required: " << std::setw(6) << ThreadConfiguration.LogicalThreadsPerBlock << std::endl;
	std::cout << "   GPU threads per block:              " << std::setw(6) << ThreadConfiguration.BlockSize << std::endl;
	std::cout << "   Number of block launches:           " << std::setw(6) << ThreadConfiguration.NumberOfBlockLaunches << std::endl;
	std::cout << "   Thread padding:                     " << std::setw(6) << ThreadConfiguration.ThreadPaddingPerBlock << std::endl;
	std::cout << "   GridSize (total number of blocks):  " << std::setw(6) << ThreadConfiguration.GridSize << std::endl;
	std::cout << "   Total logical threads required:     " << std::setw(6) << ThreadConfiguration.TotalLogicalThreads << std::endl;
	std::cout << "   Thread efficinecy:                  " << std::setw(6) << (double)(NS*UPS)/ThreadConfiguration.TotalLogicalThreads << std::endl;
	std::cout << "   Number of idle logical threads:     " << std::setw(6) << ThreadConfiguration.TotalLogicalThreads - (NS*UPS) << std::endl << std::endl;
	
	
	// GLOBAL MEMORY MANAGEMENT
	std::cout << "GLOBAL MEMORY MANAGEMENT:" << std::endl;
	
	SizeOfTimeDomain               = (long) NS * 2;
	SizeOfActualState              = (long) ThreadConfiguration.TotalLogicalThreads * UD;
	SizeOfActualTime               = (long) NS;
	SizeOfUnitParameters           = (long) ThreadConfiguration.TotalLogicalThreads * NUP;
	SizeOfSystemParameters         = (long) NS * NSP;
	SizeOfGlobalParameters         = (long) NGP;
	SizeOfIntegerGlobalParameters  = (long) NiGP;
	SizeOfUnitAccessories          = (long) ThreadConfiguration.TotalLogicalThreads * NUA;
	SizeOfIntegerUnitAccessories   = (long) ThreadConfiguration.TotalLogicalThreads * NiUA;
	SizeOfSystemAccessories        = (long) NS * NSA;
	SizeOfIntegerSystemAccessories = (long) NS * NiSA;
	SizeOfEvents                   = (long) ThreadConfiguration.TotalLogicalThreads * NE;
	SizeOfCouplingStrength         = (long) NC * NS;
	SizeOfCouplingIndex            = (long) NC;
	SizeOfDenseOutputIndex         = (long) NS;
	SizeOfDenseOutputTimeInstances = (long) NS * NDO;
	SizeOfDenseOutputStates        = (long) ThreadConfiguration.TotalLogicalThreads * UD * NDO;
	
	SizeOfCouplingMatrixOriginal   = (long) NC * UPS * UPS;
	
	if ( ( CCI == 0 ) && ( CBW == 0 ) ) // Full irregular
		SizeOfCouplingMatrix = (long) NC * UPS * UPS;
	
	if ( ( CCI == 0 ) && ( CBW > 0  ) ) // Diagonal (including the circular extensions)
		SizeOfCouplingMatrix = (long) NC * UPS * (2*CBW+1);
	
	if ( ( CCI == 1 ) && ( CBW > 0  ) ) // Circularly diagonal
		SizeOfCouplingMatrix = (long) NC * (2*CBW+1);
	
	if ( ( CCI == 1 ) && ( CBW == 0 ) ) // Full circular
		SizeOfCouplingMatrix = (long) NC * UPS;
	
	GlobalMemoryRequired = sizeof(Precision) * ( SizeOfTimeDomain + \
												 SizeOfActualState + \
												 SizeOfUnitParameters + \
												 SizeOfSystemParameters + \
												 SizeOfGlobalParameters + \
												 SizeOfUnitAccessories + \
												 SizeOfSystemAccessories + \
												 SizeOfCouplingMatrix + \
												 SizeOfDenseOutputTimeInstances + \
												 SizeOfDenseOutputStates + \
												 2*UD + NE) + \
						   sizeof(int) * ( SizeOfIntegerGlobalParameters + \
										   SizeOfIntegerUnitAccessories + \
										   SizeOfIntegerSystemAccessories + \
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
	h_UnitParameters           = AllocateHostPinnedMemory<Precision>( SizeOfUnitParameters );
	h_SystemParameters         = AllocateHostPinnedMemory<Precision>( SizeOfSystemParameters );
	h_GlobalParameters         = AllocateHostPinnedMemory<Precision>( SizeOfGlobalParameters );
	h_IntegerGlobalParameters  = AllocateHostPinnedMemory<int>( SizeOfIntegerGlobalParameters );
	h_UnitAccessories          = AllocateHostPinnedMemory<Precision>( SizeOfUnitAccessories );
	h_IntegerUnitAccessories   = AllocateHostPinnedMemory<int>( SizeOfIntegerUnitAccessories );
	h_SystemAccessories        = AllocateHostPinnedMemory<Precision>( SizeOfSystemAccessories );
	h_IntegerSystemAccessories = AllocateHostPinnedMemory<int>( SizeOfIntegerSystemAccessories );
	h_CouplingMatrix           = AllocateHostPinnedMemory<Precision>( SizeOfCouplingMatrix );
	h_CouplingMatrixOriginal   = AllocateHostPinnedMemory<Precision>( SizeOfCouplingMatrixOriginal );
	h_CouplingStrength         = AllocateHostPinnedMemory<Precision>( SizeOfCouplingStrength );
	h_CouplingIndex            = AllocateHostPinnedMemory<int>( SizeOfCouplingIndex );
	h_DenseOutputIndex         = AllocateHostPinnedMemory<int>( SizeOfDenseOutputIndex );
	h_DenseOutputTimeInstances = AllocateHostPinnedMemory<Precision>( SizeOfDenseOutputTimeInstances );
	h_DenseOutputStates        = AllocateHostPinnedMemory<Precision>( SizeOfDenseOutputStates );
	
	GlobalVariables.d_TimeDomain               = AllocateDeviceMemory<Precision>( SizeOfTimeDomain );               // SHARED
	GlobalVariables.d_ActualState              = AllocateDeviceMemory<Precision>( SizeOfActualState );              // REGISTERS
	GlobalVariables.d_ActualTime               = AllocateDeviceMemory<Precision>( SizeOfActualTime );               // SHARED
	GlobalVariables.d_UnitParameters           = AllocateDeviceMemory<Precision>( SizeOfUnitParameters );           // REGISTERS
	GlobalVariables.d_SystemParameters         = AllocateDeviceMemory<Precision>( SizeOfSystemParameters );         // SHARED
	GlobalVariables.d_GlobalParameters         = AllocateDeviceMemory<Precision>( SizeOfGlobalParameters );         // SHARED/GLOBAL
	GlobalVariables.d_IntegerGlobalParameters  = AllocateDeviceMemory<int>( SizeOfIntegerGlobalParameters );        // SHARED/GLOBAL
	GlobalVariables.d_UnitAccessories          = AllocateDeviceMemory<Precision>( SizeOfUnitAccessories );          // REGISTERS
	GlobalVariables.d_IntegerUnitAccessories   = AllocateDeviceMemory<int>( SizeOfIntegerUnitAccessories );         // REGISTERS
	GlobalVariables.d_SystemAccessories        = AllocateDeviceMemory<Precision>( SizeOfSystemAccessories );        // SHARED
	GlobalVariables.d_IntegerSystemAccessories = AllocateDeviceMemory<int>( SizeOfIntegerSystemAccessories );       // SHARED
	GlobalVariables.d_CouplingMatrix           = AllocateDeviceMemory<Precision>( SizeOfCouplingMatrix );           // SHARED/GLOBAL
	GlobalVariables.d_CouplingStrength         = AllocateDeviceMemory<Precision>( SizeOfCouplingStrength );         // SHARED
	GlobalVariables.d_CouplingIndex            = AllocateDeviceMemory<int>( SizeOfCouplingIndex );                  // SHARED
	GlobalVariables.d_DenseOutputIndex         = AllocateDeviceMemory<int>( SizeOfDenseOutputIndex );               // SHARED
	GlobalVariables.d_DenseOutputTimeInstances = AllocateDeviceMemory<Precision>( SizeOfDenseOutputTimeInstances ); // GLOBAL
	GlobalVariables.d_DenseOutputStates        = AllocateDeviceMemory<Precision>( SizeOfDenseOutputStates );        // GLOBAL
	GlobalVariables.d_RelativeTolerance        = AllocateDeviceMemory<Precision>( UD );                             // SHARED(ADAPTIVE)
	GlobalVariables.d_AbsoluteTolerance        = AllocateDeviceMemory<Precision>( UD );                             // SHARED(ADAPTIVE)
	GlobalVariables.d_EventTolerance           = AllocateDeviceMemory<Precision>( NE );                             // SHARED
	GlobalVariables.d_EventDirection           = AllocateDeviceMemory<int>( NE );                                   // SHARED
	
	
	// SHARED MEMORY MANAGEMENT
	std::cout << "SHARED MEMORY MANAGEMENT:" << std::endl;
	
	SharedMemoryUsage.SingleCouplingMatrixSize = SizeOfCouplingMatrix/NC;
	
	SharedMemoryUsage.GlobalVariables  = 1; // Default: ON
	SharedMemoryUsage.CouplingMatrices = 0; // Default: OFF
	
	if ( CCI == 1 )
	{
		SharedMemoryUsage.CouplingMatrices = 1; // Circular matrix: ON
		std::cout << "   Coupling matrix is circular: its shared memory usage is automatically ON!" << std::endl << std::endl;
	}
	
	switch (Algorithm)
	{
		case RK4:
			SharedMemoryUsage.IsAdaptive = 0;
			break;
		default:
			SharedMemoryUsage.IsAdaptive = 1;
			break;
	}
	
	SharedMemoryRequiredGlobalVariables  = sizeof(Precision)*(NGP) + sizeof(int)*(NiGP);
	SharedMemoryRequiredCouplingMatrices = sizeof(Precision)*(SizeOfCouplingMatrix);
	
	DynamicSharedMemoryRequired = SharedMemoryUsage.GlobalVariables  * SharedMemoryRequiredGlobalVariables + \
						          SharedMemoryUsage.CouplingMatrices * SharedMemoryRequiredCouplingMatrices;
	
	// Bank conflict if NCmod = 0, 2, 4, 8 or 16
	int NCp   = ( NC==0 ? NC+1 : ( NC==2 ? NC+1 : ( NC==4 ? NC+1 : ( NC==8 ? NC+1 : ( NC==16 ? NC+1 : NC ) ) ) ) );
	// Bank conflicts if NSP, NSA and NiSA = 4, 8 or 16
	int NSPp  = (  NSP==4 ?  NSP+1 : (  NSP==8 ?  NSP+1 : (  NSP==16 ?  NSP+1 : NSP  ) ) );
	int NSAp  = (  NSA==4 ?  NSA+1 : (  NSA==8 ?  NSA+1 : (  NSA==16 ?  NSA+1 : NSA  ) ) );
	int NiSAp = ( NiSA==4 ? NiSA+1 : ( NiSA==8 ? NiSA+1 : ( NiSA==16 ? NiSA+1 : NiSA ) ) );
	
	StaticSharedMemoryRequired = sizeof(Precision)*( SPB*UPS*NCp ) + \									// Solver
	                             sizeof(Precision)*( SPB*NCp ) + \										// Solver
								 sizeof(Precision)*( SPB*2 ) + \										// Solver
								 sizeof(Precision)*( SPB ) + \											// Solver
								 sizeof(Precision)*( SPB ) + \											// Solver
								 sizeof(Precision)*( SPB ) + \											// Solver
								 sizeof(Precision)*( (NSPp==0 ? 1 : SPB) * (NSPp==0 ? 1 : NSPp) ) + \	// Solver
								 sizeof(Precision)*( (NSAp==0 ? 1 : SPB) * (NSAp==0 ? 1 : NSAp) ) + \	// Solver
								 sizeof(Precision)*( (SharedMemoryUsage.IsAdaptive==0 ? 1 : UD) ) + \	// Solver
								 sizeof(Precision)*( (SharedMemoryUsage.IsAdaptive==0 ? 1 : UD) ) + \	// Solver
								 sizeof(Precision)*( (NE==0 ? 1 : NE) ) + \								// Solver
								 sizeof(Precision)*( SPB ) + \											// Solver
								 sizeof(Precision)*( SPB ) + \											// Stepper
								 sizeof(Precision)*( (SharedMemoryUsage.IsAdaptive==0 ? 0 : SPB) ) + \	// ErrorController
								 sizeof(Precision)*( (SharedMemoryUsage.IsAdaptive==0 ? 0 : SPB) ) + \	// ErrorController
								 sizeof(Precision)*( (NE==0 ? 0 : SPB) ) + \							// EventHandling
								 sizeof(int)*( NC ) + \													// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( (NiSAp==0 ? 1 : SPB) * (NiSAp==0 ? 1 : NiSAp) ) + \		// Solver
								 sizeof(int)*( (NE==0 ? 1 : NE) ) + \									// Solver
								 sizeof(int)*( 1 ) + \													// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( SPB ) + \												// Solver
								 sizeof(int)*( (NE==0 ? 0 : SPB) );										// EventHandling
	
	SharedMemoryRequired = DynamicSharedMemoryRequired + StaticSharedMemoryRequired;
	
	SharedMemoryRequiredUpperLimit = StaticSharedMemoryRequired + \
	                                 SharedMemoryRequiredGlobalVariables + \
						             SharedMemoryRequiredCouplingMatrices;
	
	SharedMemoryAvailable = SelectedDeviceProperties.sharedMemPerBlock;
	
	std::cout << "   Required shared memory per block for managable variables:" << std::endl;
	std::cout << "    Shared memory required by global variables:           " << std::setw(6) << SharedMemoryRequiredGlobalVariables << " b (" << ( (SharedMemoryUsage.GlobalVariables     == 0) ? "OFF" : "ON " ) << " -> SolverOption)" << std::endl;
	std::cout << "    Shared memory required by coupling matrices:          " << std::setw(6) << SharedMemoryRequiredCouplingMatrices << " b (" << ( (SharedMemoryUsage.CouplingMatrices    == 0) ? "OFF" : "ON " ) << " -> SolverOption)" << std::endl;
	std::cout << "   Upper limit of possible shared memory usage per block: " << std::setw(6) << SharedMemoryRequiredUpperLimit << " b (Internals + All is ON)" << std::endl;
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
	SolverOptions.ActiveSystems              = NS;
	SolverOptions.MaximumTimeStep            = 1.0e6;
	SolverOptions.MinimumTimeStep            = 1.0e-12;
	SolverOptions.TimeStepGrowLimit          = 5.0;
	SolverOptions.TimeStepShrinkLimit        = 0.1;
	SolverOptions.DenseOutputMinimumTimeStep = 0.0;
	SolverOptions.DenseOutputSaveFrequency   = 1;
	
	Precision DefaultAlgorithmTolerances = 1e-8;
	for (int i=0; i<UD; i++)
	{
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_RelativeTolerance+i, &DefaultAlgorithmTolerances, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_AbsoluteTolerance+i, &DefaultAlgorithmTolerances, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
	}
	
	Precision DefaultEventTolerance = 1e-6;
	int       DefaultEventDirection = 0;
	for ( int i=0; i< NE; i++ )
	{
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventTolerance+i,   &DefaultEventTolerance, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventDirection+i,   &DefaultEventDirection, sizeof(int),       cudaMemcpyHostToDevice, Stream) );
	}
	
	std::cout << "   Initial time step:                        " << std::setw(6) << SolverOptions.InitialTimeStep << std::endl;
	std::cout << "   Active systems:                           " << std::setw(6) << SolverOptions.ActiveSystems << std::endl;
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
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::~ProblemSolver()
{
    gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaStreamDestroy(Stream) );
	gpuErrCHK( cudaEventDestroy(Event) );
	
	gpuErrCHK( cudaFreeHost(h_TimeDomain) );
	gpuErrCHK( cudaFreeHost(h_ActualState) );
	gpuErrCHK( cudaFreeHost(h_ActualTime) );
	gpuErrCHK( cudaFreeHost(h_UnitParameters) );
	gpuErrCHK( cudaFreeHost(h_SystemParameters) );
	gpuErrCHK( cudaFreeHost(h_GlobalParameters) );
	gpuErrCHK( cudaFreeHost(h_IntegerGlobalParameters) );
	gpuErrCHK( cudaFreeHost(h_UnitAccessories) );
	gpuErrCHK( cudaFreeHost(h_IntegerUnitAccessories) );
	gpuErrCHK( cudaFreeHost(h_SystemAccessories) );
	gpuErrCHK( cudaFreeHost(h_IntegerSystemAccessories) );
	gpuErrCHK( cudaFreeHost(h_CouplingMatrix) );
	gpuErrCHK( cudaFreeHost(h_CouplingMatrixOriginal) );
	gpuErrCHK( cudaFreeHost(h_CouplingStrength) );
	gpuErrCHK( cudaFreeHost(h_CouplingIndex) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputIndex) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputStates) );
	
	gpuErrCHK( cudaFree(GlobalVariables.d_TimeDomain) );
	gpuErrCHK( cudaFree(GlobalVariables.d_ActualState) );
	gpuErrCHK( cudaFree(GlobalVariables.d_ActualTime) );
	gpuErrCHK( cudaFree(GlobalVariables.d_UnitParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_SystemParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_GlobalParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerGlobalParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_UnitAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerUnitAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_SystemAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerSystemAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_CouplingMatrix) );
	gpuErrCHK( cudaFree(GlobalVariables.d_CouplingStrength) );
	gpuErrCHK( cudaFree(GlobalVariables.d_CouplingIndex) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputIndex) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputStates) );
	gpuErrCHK( cudaFree(GlobalVariables.d_RelativeTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_AbsoluteTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventDirection) );
	
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Object for Parameters scan is deleted!" << std::endl;
	std::cout << "Every memory have been deallocated!" << std::endl;
	std::cout << "Coo man coo!!!" << std::endl;
	std::cout << "--------------------------------------" << std::endl << std::endl;
}

// BOUND CHECK, set/get host, options
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::BoundCheck(std::string Function, std::string Variable, int Value, int LowerLimit, int UpperLimit)
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
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SharedMemoryCheck()
{
	std::cout << "SHARED MEMORY MANAGEMENT CHANGED BY SOLVER OPTION:" << std::endl;
	
	if ( CCI == 1 )
	{
		SharedMemoryUsage.CouplingMatrices = 1; // Circular matrix: ON
		std::cout << "   Coupling matrix is circular: its shared memory usage is automatically ON!" << std::endl << std::endl;
	}
	
	DynamicSharedMemoryRequired = SharedMemoryUsage.GlobalVariables  * SharedMemoryRequiredGlobalVariables + \
						          SharedMemoryUsage.CouplingMatrices * SharedMemoryRequiredCouplingMatrices;
	
	SharedMemoryRequired = DynamicSharedMemoryRequired + StaticSharedMemoryRequired;
	
	std::cout << "   Required shared memory per block for managable variables:" << std::endl;
	std::cout << "    Shared memory required by global variables:           " << std::setw(6) << SharedMemoryRequiredGlobalVariables << " b (" << ( (SharedMemoryUsage.GlobalVariables     == 0) ? "OFF" : "ON " ) << " -> SolverOption)" << std::endl;
	std::cout << "    Shared memory required by coupling matrices:          " << std::setw(6) << SharedMemoryRequiredCouplingMatrices << " b (" << ( (SharedMemoryUsage.CouplingMatrices    == 0) ? "OFF" : "ON " ) << " -> SolverOption)" << std::endl;
	std::cout << "   Upper limit of possible shared memory usage per block: " << std::setw(6) << SharedMemoryRequiredUpperLimit << " b (Internals + All is ON)" << std::endl;
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
}

// OPTION, single input argument
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, T Value)
{
	switch (Option)
	{
		case InitialTimeStep:
			SolverOptions.InitialTimeStep = (Precision)Value;
			break;
		
		case ActiveSystems:
			SolverOptions.ActiveSystems = (int)Value;
			break;
		
		case MaximumTimeStep:
			SolverOptions.MaximumTimeStep = (Precision)Value;
			break;
		
		case MinimumTimeStep:
			SolverOptions.MinimumTimeStep = (Precision)Value;
			break;
		
		case TimeStepGrowLimit:
			SolverOptions.TimeStepGrowLimit = (Precision)Value;
			break;
		
		case TimeStepShrinkLimit:
			SolverOptions.TimeStepShrinkLimit = (Precision)Value;
			break;
		
		case DenseOutputMinimumTimeStep:
			SolverOptions.DenseOutputMinimumTimeStep = (Precision)Value;
			break;
		
		case DenseOutputSaveFrequency:
			SolverOptions.DenseOutputSaveFrequency = (int)Value;
			break;
		//---------------------------------------
		case SharedGlobalVariables:
			SharedMemoryUsage.GlobalVariables = (bool)Value;
			SharedMemoryCheck();
			break;
		
		case SharedCouplingMatrices:
			SharedMemoryUsage.CouplingMatrices = (bool)Value;
			SharedMemoryCheck();
			break;
		
		//---------------------------------------
		default:
			std::cerr << "ERROR: In solver member function SolverOption!" << std::endl;
			std::cerr << "       Option: " << SolverOptionsToString(Option) << std::endl;
			std::cerr << "       This option needs 2 input arguments!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// OPTION, double input argument
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, int Index, T Value)
{
	Precision PValue = (Precision)Value;
	int       IValue = (int)Value;
	switch (Option)
	{
		case RelativeTolerance:
			BoundCheck("SolverOption", "RelativeTolerance", Index, 0, UD-1);
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_RelativeTolerance+Index, &PValue, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case AbsoluteTolerance:
			BoundCheck("SolverOption", "AbsoluteTolerance", Index, 0, UD-1);
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
			std::cerr << "       This option needs 1 input arguments!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Unit scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SetHost(int SystemNumber, int UnitNumber, ListOfVariables Variable, int SerialNumber, T Value)
{
	BoundCheck("SetHost", "SystemNumber", SystemNumber, 0, NS-1 );
	BoundCheck("SetHost", "UnitNumber",   UnitNumber,   0, UPS-1);
	
	int BlockID       = SystemNumber / SPB;
	int LocalSystemID = SystemNumber % SPB;
	
	int GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitNumber;
	int GlobalMemoryID         = GlobalThreadID_Logical + SerialNumber*ThreadConfiguration.TotalLogicalThreads;
	
	switch (Variable)
	{
		case ActualState:
			BoundCheck("SetHost", "ActualState", SerialNumber, 0, UD-1);
			h_ActualState[GlobalMemoryID] = (Precision)Value;
			break;
		
		case UnitParameters:
			BoundCheck("SetHost", "UnitParameters", SerialNumber, 0, NUP-1);
			h_UnitParameters[GlobalMemoryID] = (Precision)Value;
			break;
		
		case UnitAccessories:
			BoundCheck("SetHost", "UnitAccessories", SerialNumber, 0, NUA-1);
			h_UnitAccessories[GlobalMemoryID] = (Precision)Value;
			break;
		
		case IntegerUnitAccessories:
			BoundCheck("SetHost", "IntegerUnitAccessories", SerialNumber, 0, NiUA-1);
			h_IntegerUnitAccessories[GlobalMemoryID] = (int)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, System scope and dense time
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SetHost(int SystemNumber, ListOfVariables Variable, int SerialNumber, T Value)
{
	BoundCheck("SetHost", "SystemNumber", SystemNumber, 0, NS-1);
	
	int GlobalSystemID = SystemNumber;
	int GlobalMemoryID = GlobalSystemID + SerialNumber*NS;
	
	switch (Variable)
	{
		case TimeDomain:
			BoundCheck("SetHost", "TimeDomain", SerialNumber, 0, 1);
			h_TimeDomain[GlobalMemoryID] = (Precision)Value;
			break;
		
		case ActualTime:
			BoundCheck("SetHost", "ActualTime", SerialNumber, 0, 0);
			h_ActualTime[GlobalMemoryID] = (Precision)Value;
			break;
		
		case SystemParameters:
			BoundCheck("SetHost", "SystemParameters", SerialNumber, 0, NSP-1);
			h_SystemParameters[GlobalMemoryID] = (Precision)Value;
			break;
		
		case SystemAccessories:
			BoundCheck("SetHost", "SystemAccessories", SerialNumber, 0, NSA-1);
			h_SystemAccessories[GlobalMemoryID] = (Precision)Value;
			break;
		
		case IntegerSystemAccessories:
			BoundCheck("SetHost", "IntegerSystemAccessories", SerialNumber, 0, NiSA-1);
			h_IntegerSystemAccessories[GlobalMemoryID] = (int)Value;
			break;
		
		case DenseTime:
			BoundCheck("SetHost", "DenseTime", SerialNumber, 0, NDO-1);
			h_DenseOutputTimeInstances[GlobalMemoryID] = (Precision)Value;
			break;
		
		case CouplingStrength:
			BoundCheck("SetHost", "CouplingStrength", SerialNumber, 0, NC-1);
			h_CouplingStrength[GlobalMemoryID] = (Precision)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Global scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SetHost(ListOfVariables Variable, int SerialNumber, T Value)
{
	int GlobalMemoryID = SerialNumber;
	
	switch (Variable)
	{
		case GlobalParameters:
			BoundCheck("SetHost", "GlobalParameters", SerialNumber, 0, NGP-1);
			h_GlobalParameters[GlobalMemoryID] = (Precision)Value;
			break;
		
		case IntegerGlobalParameters:
			BoundCheck("SetHost", "IntegerGlobalParameters", SerialNumber, 0, NiGP-1);
			h_IntegerGlobalParameters[GlobalMemoryID] = (int)Value;
			break;
		
		case CouplingIndex:
			BoundCheck("SetHost", "CouplingIndex", SerialNumber, 0, NC-1);
			h_CouplingIndex[GlobalMemoryID] = (int)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Coupling matrix
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SetHost(int CouplingNumber, ListOfVariables Variable, int Row, int Col, T Value)
{
	BoundCheck("SetHost", "CouplingNumber", CouplingNumber, 0, NC-1);
	BoundCheck("SetHost", "Coupling Matrix Row", Row, 0, UPS-1);
	BoundCheck("SetHost", "Coupling Matrix Col", Col, 0, UPS-1);
	
	int GlobalMemoryIDOriginal = Row + Col*UPS + CouplingNumber*UPS*UPS;
	
	int ModRow;
	int ModCol;
	int ColLimit;
	int GlobalMemoryID;
	
	// Full irregular
	if ( ( CCI == 0 ) && ( CBW == 0 ) )
		GlobalMemoryID = GlobalMemoryIDOriginal;
	
	// Diagonal (including the circular extensions)
	if ( ( CCI == 0 ) && ( CBW > 0  ) ) 
	{
		ModRow = Row;
		ModCol = Col - Row + CBW;
		
		if ( ModCol >= UPS )
			ModCol = ModCol - UPS;
		
		if ( ModCol < 0 )
			ModCol = ModCol + UPS;
		
		ColLimit = 2*CBW+1;
		
		BoundCheck("SetHost", "Coupling Matrix Shifted Diagonal", ModCol, 0, ColLimit-1);
		GlobalMemoryID = ModRow + ModCol*UPS + CouplingNumber*UPS*ColLimit;
	}
	
	// Circularly diagonal
	if ( ( CCI == 1 ) && ( CBW > 0  ) )
	{
		ModRow = Row;
		ModCol = Col - Row + CBW;
		
		if ( ModCol >= UPS )
			ModCol = ModCol - UPS;
		
		if ( ModCol < 0 )
			ModCol = ModCol + UPS;
		
		ColLimit = 2*CBW+1;
		
		BoundCheck("SetHost", "Coupling Matrix Shifted Diagonal", ModCol, 0, ColLimit-1);
		GlobalMemoryID = ModCol + CouplingNumber*ColLimit;
	}
	
	// Full circular
	if ( ( CCI == 1 ) && ( CBW == 0 ) )
	{
		ModRow = Row;
		ModCol = Col - Row + CBW;
		
		if ( ModCol >= UPS )
			ModCol = ModCol - UPS;
		
		if ( ModCol < 0 )
			ModCol = ModCol + UPS;
		
		ColLimit = UPS;
		
		BoundCheck("SetHost", "Coupling Matrix Shifted Diagonal", ModCol, 0, ColLimit-1);
		GlobalMemoryID = ModCol + CouplingNumber*ColLimit;
	}
	
	
	switch (Variable)
	{
		case CouplingMatrix:
			h_CouplingMatrixOriginal[GlobalMemoryIDOriginal] = (Precision)Value;
			h_CouplingMatrix[GlobalMemoryID] = (Precision)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Dense index
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SetHost(int SystemNumber, ListOfVariables Variable, T Value)
{
	BoundCheck("SetHost", "SystemNumber", SystemNumber, 0, NS-1);
	
	int GlobalMemoryID = SystemNumber;
	
	switch (Variable)
	{
		case DenseIndex:
			h_DenseOutputIndex[GlobalMemoryID] = (int)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SETHOST, Dense state
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SetHost(int SystemNumber, int UnitNumber, ListOfVariables Variable, int ComponentNumber, int SerialNumber, T Value)
{
	BoundCheck("SetHost", "SystemNumber", SystemNumber, 0, NS-1 );
	BoundCheck("SetHost", "UnitNumber",   UnitNumber,   0, UPS-1);
	
	BoundCheck("SetHost", "ComponentNumber in dense state", ComponentNumber, 0, UD-1);
	
	int BlockID       = SystemNumber / SPB;
	int LocalSystemID = SystemNumber % SPB;
	
	int GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitNumber;
	int GlobalMemoryID         = GlobalThreadID_Logical + ComponentNumber*ThreadConfiguration.TotalLogicalThreads + SerialNumber*SizeOfActualState;
	
	switch (Variable)
	{
		case DenseState:
			BoundCheck("SetHost", "DenseState", SerialNumber, 0, NDO-1);
			h_DenseOutputStates[GlobalMemoryID] = (Precision)Value;
			break;
		
		default:
			std::cerr << "ERROR: In solver member function SetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SYNCHRONISE, Host -> Device
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SynchroniseFromHostToDevice(ListOfVariables Variable)
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
		
		case UnitParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_UnitParameters, h_UnitParameters, SizeOfUnitParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case SystemParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_SystemParameters, h_SystemParameters, SizeOfSystemParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case GlobalParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_GlobalParameters, h_GlobalParameters, SizeOfGlobalParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case IntegerGlobalParameters:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerGlobalParameters, h_IntegerGlobalParameters, SizeOfIntegerGlobalParameters*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case UnitAccessories:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_UnitAccessories, h_UnitAccessories, SizeOfUnitAccessories*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case IntegerUnitAccessories:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerUnitAccessories, h_IntegerUnitAccessories, SizeOfIntegerUnitAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case SystemAccessories:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_SystemAccessories, h_SystemAccessories, SizeOfSystemAccessories*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case IntegerSystemAccessories:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerSystemAccessories, h_IntegerSystemAccessories, SizeOfIntegerSystemAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case CouplingMatrix:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_CouplingMatrix, h_CouplingMatrix, SizeOfCouplingMatrix*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case CouplingStrength:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_CouplingStrength, h_CouplingStrength, SizeOfCouplingStrength*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case CouplingIndex:
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_CouplingIndex, h_CouplingIndex, SizeOfCouplingIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
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
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_UnitParameters, h_UnitParameters, SizeOfUnitParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_SystemParameters, h_SystemParameters, SizeOfSystemParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_GlobalParameters, h_GlobalParameters, SizeOfGlobalParameters*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerGlobalParameters, h_IntegerGlobalParameters, SizeOfIntegerGlobalParameters*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_UnitAccessories, h_UnitAccessories, SizeOfUnitAccessories*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerUnitAccessories, h_IntegerUnitAccessories, SizeOfIntegerUnitAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_SystemAccessories, h_SystemAccessories, SizeOfSystemAccessories*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_IntegerSystemAccessories, h_IntegerSystemAccessories, SizeOfIntegerSystemAccessories*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_CouplingMatrix, h_CouplingMatrix, SizeOfCouplingMatrix*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_CouplingStrength, h_CouplingStrength, SizeOfCouplingStrength*sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_CouplingIndex, h_CouplingIndex, SizeOfCouplingIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
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
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SynchroniseFromDeviceToHost(ListOfVariables Variable)
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
		
		case UnitParameters:
			gpuErrCHK( cudaMemcpyAsync(h_UnitParameters, GlobalVariables.d_UnitParameters, SizeOfUnitParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case SystemParameters:
			gpuErrCHK( cudaMemcpyAsync(h_SystemParameters, GlobalVariables.d_SystemParameters, SizeOfSystemParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case GlobalParameters:
			gpuErrCHK( cudaMemcpyAsync(h_GlobalParameters, GlobalVariables.d_GlobalParameters, SizeOfGlobalParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case IntegerGlobalParameters:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerGlobalParameters, GlobalVariables.d_IntegerGlobalParameters, SizeOfIntegerGlobalParameters*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case UnitAccessories:
			gpuErrCHK( cudaMemcpyAsync(h_UnitAccessories, GlobalVariables.d_UnitAccessories, SizeOfUnitAccessories*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case IntegerUnitAccessories:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerUnitAccessories, GlobalVariables.d_IntegerUnitAccessories, SizeOfIntegerUnitAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case SystemAccessories:
			gpuErrCHK( cudaMemcpyAsync(h_SystemAccessories, GlobalVariables.d_SystemAccessories, SizeOfSystemAccessories*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case IntegerSystemAccessories:
			gpuErrCHK( cudaMemcpyAsync(h_IntegerSystemAccessories, GlobalVariables.d_IntegerSystemAccessories, SizeOfIntegerSystemAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case CouplingMatrix:
			gpuErrCHK( cudaMemcpyAsync(h_CouplingMatrix, GlobalVariables.d_CouplingMatrix, SizeOfCouplingMatrix*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case CouplingStrength:
			gpuErrCHK( cudaMemcpyAsync(h_CouplingStrength, GlobalVariables.d_CouplingStrength, SizeOfCouplingStrength*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		case CouplingIndex:
			gpuErrCHK( cudaMemcpyAsync(h_CouplingIndex, GlobalVariables.d_CouplingIndex, SizeOfCouplingIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
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
			gpuErrCHK( cudaMemcpyAsync(h_UnitParameters, GlobalVariables.d_UnitParameters, SizeOfUnitParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_SystemParameters, GlobalVariables.d_SystemParameters, SizeOfSystemParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_GlobalParameters, GlobalVariables.d_GlobalParameters, SizeOfGlobalParameters*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerGlobalParameters, GlobalVariables.d_IntegerGlobalParameters, SizeOfIntegerGlobalParameters*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_UnitAccessories, GlobalVariables.d_UnitAccessories, SizeOfUnitAccessories*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerUnitAccessories, GlobalVariables.d_IntegerUnitAccessories, SizeOfIntegerUnitAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_SystemAccessories, GlobalVariables.d_SystemAccessories, SizeOfSystemAccessories*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_IntegerSystemAccessories, GlobalVariables.d_IntegerSystemAccessories, SizeOfIntegerSystemAccessories*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_CouplingMatrix, GlobalVariables.d_CouplingMatrix, SizeOfCouplingMatrix*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_CouplingStrength, GlobalVariables.d_CouplingStrength, SizeOfCouplingStrength*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_CouplingIndex, GlobalVariables.d_CouplingIndex, SizeOfCouplingIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex, GlobalVariables.d_DenseOutputIndex, SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, GlobalVariables.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates, GlobalVariables.d_DenseOutputStates, SizeOfDenseOutputStates*sizeof(Precision), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		default:
			std::cerr << "ERROR: In solver member function SynchroniseFromDeviceToHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option is not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Unit scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::GetHost(int SystemNumber, int UnitNumber, ListOfVariables Variable, int SerialNumber)
{
	BoundCheck("GetHost", "SystemNumber", SystemNumber, 0, NS-1 );
	BoundCheck("GetHost", "UnitNumber",   UnitNumber,   0, UPS-1);
	
	int BlockID       = SystemNumber / SPB;
	int LocalSystemID = SystemNumber % SPB;
	
	int GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitNumber;
	int GlobalMemoryID         = GlobalThreadID_Logical + SerialNumber*ThreadConfiguration.TotalLogicalThreads;
	
	switch (Variable)
	{
		case ActualState:
			BoundCheck("GetHost", "ActualState", SerialNumber, 0, UD-1);
			return (T)h_ActualState[GlobalMemoryID];
		
		case UnitParameters:
			BoundCheck("GetHost", "UnitParameters", SerialNumber, 0, NUP-1);
			return (T)h_UnitParameters[GlobalMemoryID];
		
		case UnitAccessories:
			BoundCheck("GetHost", "UnitAccessories", SerialNumber, 0, NUA-1);
			return (T)h_UnitAccessories[GlobalMemoryID];
		
		case IntegerUnitAccessories:
			BoundCheck("GetHost", "IntegerUnitAccessories", SerialNumber, 0, NiUA-1);
			return (T)h_IntegerUnitAccessories[GlobalMemoryID];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, System scope and dense time
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::GetHost(int SystemNumber, ListOfVariables Variable, int SerialNumber)
{
	BoundCheck("GetHost", "SystemNumber", SystemNumber, 0, NS-1);
	
	int GlobalSystemID = SystemNumber;
	int GlobalMemoryID = GlobalSystemID + SerialNumber*NS;
	
	switch (Variable)
	{
		case TimeDomain:
			BoundCheck("GetHost", "TimeDomain", SerialNumber, 0, 1);
			return (T)h_TimeDomain[GlobalMemoryID];
		
		case ActualTime:
			BoundCheck("GetHost", "ActualTime", SerialNumber, 0, 0);
			return (T)h_ActualTime[GlobalMemoryID];
		
		case SystemParameters:
			BoundCheck("GetHost", "SystemParameters", SerialNumber, 0, NSP-1);
			return (T)h_SystemParameters[GlobalMemoryID];
		
		case SystemAccessories:
			BoundCheck("GetHost", "SystemAccessories", SerialNumber, 0, NSA-1);
			return (T)h_SystemAccessories[GlobalMemoryID];
		
		case IntegerSystemAccessories:
			BoundCheck("GetHost", "IntegerSystemAccessories", SerialNumber, 0, NiSA-1);
			return (T)h_IntegerSystemAccessories[GlobalMemoryID];
		
		case DenseTime:
			BoundCheck("GetHost", "DenseTime", SerialNumber, 0, NDO-1);
			return (T)h_DenseOutputTimeInstances[GlobalMemoryID];
		
		case CouplingStrength:
			BoundCheck("GetHost", "CouplingStrength", SerialNumber, 0, NC-1);
			return (T)h_CouplingStrength[GlobalMemoryID];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Global scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::GetHost(ListOfVariables Variable, int SerialNumber)
{
	int GlobalMemoryID = SerialNumber;
	
	switch (Variable)
	{
		case GlobalParameters:
			BoundCheck("GetHost", "GlobalParameters", SerialNumber, 0, NGP-1);
			return (T)h_GlobalParameters[GlobalMemoryID];
		
		case IntegerGlobalParameters:
			BoundCheck("GetHost", "IntegerGlobalParameters", SerialNumber, 0, NiGP-1);
			return (T)h_IntegerGlobalParameters[GlobalMemoryID];
		
		case CouplingIndex:
			BoundCheck("GetHost", "CouplingIndex", SerialNumber, 0, NC-1);
			return (T)h_CouplingIndex[GlobalMemoryID];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Coupling matrix
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::GetHost(int CouplingNumber, ListOfVariables Variable, int Row, int Col)
{
	BoundCheck("GetHost", "CouplingNumber", CouplingNumber, 0, NC-1);
	BoundCheck("GetHost", "Row", Row, 0, UPS-1);
	BoundCheck("GetHost", "Col", Col, 0, UPS-1);
	
	int GlobalMemoryID = Row + Col*UPS + CouplingNumber*UPS*UPS;
	
	switch (Variable)
	{
		case CouplingMatrix:
			return (T)h_CouplingMatrixOriginal[GlobalMemoryID];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Dense index
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::GetHost(int SystemNumber, ListOfVariables Variable)
{
	BoundCheck("GetHost", "SystemNumber", SystemNumber, 0, NS-1);
	
	int GlobalMemoryID = SystemNumber;
	
	switch (Variable)
	{
		case DenseIndex:
			return (T)h_DenseOutputIndex[GlobalMemoryID];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// GETHOST, Dense state
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
T ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::GetHost(int SystemNumber, int UnitNumber, ListOfVariables Variable, int ComponentNumber, int SerialNumber)
{
	BoundCheck("GetHost", "SystemNumber", SystemNumber, 0, NS-1 );
	BoundCheck("GetHost", "UnitNumber",   UnitNumber,   0, UPS-1);
	
	BoundCheck("GetHost", "ComponentNumber in dense state", ComponentNumber, 0, UD-1);
	
	int BlockID       = SystemNumber / SPB;
	int LocalSystemID = SystemNumber % SPB;
	
	int GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + UnitNumber;
	int GlobalMemoryID         = GlobalThreadID_Logical + ComponentNumber*ThreadConfiguration.TotalLogicalThreads + SerialNumber*SizeOfActualState;
	
	switch (Variable)
	{
		case DenseState:
			BoundCheck("GetHost", "DenseState", SerialNumber, 0, NDO-1);
			return (T)h_DenseOutputStates[GlobalMemoryID];
		
		default:
			std::cerr << "ERROR: In solver member function GetHost!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// WRITE TO FILE, Unit scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::WriteToFileUniteScope(std::string FileName, int NumberOfRows, int NumberOfColumns, T* Data)
{
	std::ofstream DataFile;
	DataFile.open (FileName);
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int TotalLogicalThreadsPerBlock = ThreadConfiguration.LogicalThreadsPerBlock + ThreadConfiguration.ThreadPaddingPerBlock;
	int GlobalThreadID_Logical;
	int GlobalMemoryID;
	int LocalThreadID_Logical;
	int LocalThreadLimit_Logical;
	int GlobalThreadLimit_Logical;
	
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			GlobalThreadID_Logical = i;
			GlobalMemoryID         = i + j*NumberOfRows;
			LocalThreadID_Logical  = i % (TotalLogicalThreadsPerBlock);
			
			LocalThreadLimit_Logical  = ThreadConfiguration.LogicalThreadsPerBlock;
			GlobalThreadLimit_Logical = (NS / SPB) * (TotalLogicalThreadsPerBlock) + (NS % SPB) * UPS;
			
			if ( ( LocalThreadID_Logical < LocalThreadLimit_Logical ) && ( GlobalThreadID_Logical < GlobalThreadLimit_Logical ) )
			{
				DataFile.width(Width); DataFile << Data[GlobalMemoryID] << ',';
			} else
			{
				DataFile.width(Width); DataFile << "PADDING" << ',';
			}
		}
		DataFile << '\n';
	}
	DataFile.close();
}

// WRITE TO FILE, System and global scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::WriteToFileSystemAndGlobalScope(std::string FileName, int NumberOfRows, int NumberOfColumns, T* Data)
{
	std::ofstream DataFile;
	DataFile.open (FileName);
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int GlobalMemoryID;
	
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			GlobalMemoryID = i + j*NumberOfRows;
			DataFile.width(Width); DataFile << Data[GlobalMemoryID] << ',';
		}
		DataFile << '\n';
	}
	DataFile.close();
}

// WRITE TO FILE, Dense output
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::WriteToFileDenseOutput(std::string FileName, int NumberOfRows, int SystemNumber, T* TimeInstances, T* States)
{
	std::ofstream DataFile;
	DataFile.open (FileName);
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(std::ios::scientific);
	
	int GlobalMemoryID;
	int GlobalThreadID_Logical;
	
	int GlobalSystemID = SystemNumber;
	int BlockID        = SystemNumber / SPB;
	int LocalSystemID  = SystemNumber % SPB;
	
	std::string String;
	
	
	DataFile << "UnitParameters:\n";
	DataFile.width(Width); DataFile << "Serial No." << ',';
	for (int i=0; i<UPS; i++) // Loop over units
	{
		DataFile.width(Width); DataFile << "U" + std::to_string(i);
		
		if ( i<(UPS-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	for (int i=0; i<NUP; i++) // Loop over unit parameters
	{
		DataFile.width(Width); DataFile << i << ',';
		for (int j=0; j<UPS; j++) // Loop over units
		{
			GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + j;
			GlobalMemoryID         = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads;
			
			DataFile.width(Width); DataFile << h_UnitParameters[GlobalMemoryID];
			
			if ( j<(UPS-1) )
				DataFile << ',';
		}
		DataFile << '\n';
	}
	DataFile << '\n';
	
	
	DataFile << "UnitAccessories:\n";
	DataFile.width(Width); DataFile << "Serial No." << ',';
	for (int i=0; i<UPS; i++) // Loop over units
	{
		DataFile.width(Width); DataFile << "U" + std::to_string(i);
		
		if ( i<(UPS-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	for (int i=0; i<NUA; i++) // Loop over unit accessories
	{
		DataFile.width(Width); DataFile << i << ',';
		for (int j=0; j<UPS; j++) // Loop over units
		{
			GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + j;
			GlobalMemoryID         = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads;
			
			DataFile.width(Width); DataFile << h_UnitAccessories[GlobalMemoryID];
			
			if ( j<(UPS-1) )
				DataFile << ',';
		}
		DataFile << '\n';
	}
	DataFile << '\n';
	
	
	DataFile << "IntegerUnitAccessories:\n";
	DataFile.width(Width); DataFile << "Serial No." << ',';
	for (int i=0; i<UPS; i++) // Loop over units
	{
		DataFile.width(Width); DataFile << "U" + std::to_string(i);
		
		if ( i<(UPS-1) )
			DataFile << ',';
	}
	DataFile << '\n';
	
	for (int i=0; i<NiUA; i++) // Loop over integer unit accessories
	{
		DataFile.width(Width); DataFile << i << ',';
		for (int j=0; j<UPS; j++) // Loop over units
		{
			GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + j;
			GlobalMemoryID         = GlobalThreadID_Logical + i*ThreadConfiguration.TotalLogicalThreads;
			
			DataFile.width(Width); DataFile << h_IntegerUnitAccessories[GlobalMemoryID];
			
			if ( j<(UPS-1) )
				DataFile << ',';
		}
		DataFile << '\n';
	}
	DataFile << '\n';
	
	
	DataFile << "SystemParameters:\n";
	for (int i=0; i<NSP; i++) // Loop over system parameters
	{
		GlobalMemoryID = GlobalSystemID + i*NS;
		
		DataFile.width(Width); DataFile << h_SystemParameters[GlobalMemoryID];
		
		if ( i<(NSP-1) )
			DataFile << ',';
	}
	DataFile << '\n' << '\n';
	
	
	DataFile << "SystemAccessories:\n";
	for (int i=0; i<NSA; i++) // Loop over system accessories
	{
		GlobalMemoryID = GlobalSystemID + i*NS;
		
		DataFile.width(Width); DataFile << h_SystemAccessories[GlobalMemoryID];
		
		if ( i<(NSA-1) )
			DataFile << ',';
	}
	DataFile << '\n' << '\n';
	
	
	DataFile << "IntegerSystemAccessories:\n";
	for (int i=0; i<NiSA; i++) // Loop over system integer accessories
	{
		GlobalMemoryID = GlobalSystemID + i*NS;
		
		DataFile.width(Width); DataFile << h_IntegerSystemAccessories[GlobalMemoryID];
		
		if ( i<(NiSA-1) )
			DataFile << ',';
	}
	DataFile << '\n' << '\n';
	
	
	DataFile << "GlobalParameters:\n";
	for (int i=0; i<NGP; i++) // Loop over global parameters
	{
		DataFile.width(Width); DataFile << h_GlobalParameters[i];
		
		if ( i<(NGP-1) )
			DataFile << ',';
	}
	DataFile << '\n' << '\n';
	
	
	DataFile << "IntegerGlobalParameters:\n";
	for (int i=0; i<NiGP; i++) // Loop over integer global parameters
	{
		DataFile.width(Width); DataFile << h_IntegerGlobalParameters[i];
		
		if ( i<(NiGP-1) )
			DataFile << ',';
	}
	DataFile << '\n' << '\n';
	
	
	DataFile << "Time series:\n";
	DataFile.width(Width); DataFile << "t" << ',';
	for (int j=0; j<UPS; j++) // Loop over units
	{
		for (int k=0; k<UD; k++) // Loop over components
		{
			DataFile.width(Width); DataFile << "U" + std::to_string(j) + "/C" + std::to_string(k);
			
			if ( ( j<(UPS-1) ) || ( k<(UD-1) ) )
				DataFile << ',';
		}
	}
	DataFile << '\n';
	
	for (int i=0; i<NumberOfRows; i++) // Loop over time steps
	{
		GlobalMemoryID = GlobalSystemID + i*NS;
		DataFile.width(Width); DataFile << TimeInstances[GlobalMemoryID] << ',';
		
		for (int j=0; j<UPS; j++) // Loop over units
		{
			for (int k=0; k<UD; k++) // Loop over components
			{
				GlobalThreadID_Logical = BlockID*(ThreadConfiguration.LogicalThreadsPerBlock+ThreadConfiguration.ThreadPaddingPerBlock) + LocalSystemID*UPS + j;
				GlobalMemoryID         = GlobalThreadID_Logical + k*ThreadConfiguration.TotalLogicalThreads + i*SizeOfActualState;
				
				DataFile.width(Width); DataFile << States[GlobalMemoryID];
				
				if ( ( j<(UPS-1) ) || ( k<(UD-1) ) )
					DataFile << ',';
			}
		}
		DataFile << '\n';
	}
	DataFile.close();
}

// PRINT, Unit, system and global scope
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::Print(ListOfVariables Variable)
{
	std::string FileName;
	int NumberOfRows;
	int NumberOfColumns;
	
	switch (Variable)
	{
		case TimeDomain:
			FileName        = "TimeDomainInSolverObject.txt";
			NumberOfRows    = NS;
			NumberOfColumns = 2;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_TimeDomain);
			break;
		
		case ActualTime:
			FileName        = "ActualTimeInSolverObject.txt";
			NumberOfRows    = NS;
			NumberOfColumns = 1;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_ActualTime);
			break;
		
		case ActualState:
			FileName        = "ActualStateInSolverObject.txt";
			NumberOfRows    = ThreadConfiguration.TotalLogicalThreads;
			NumberOfColumns = UD;
			WriteToFileUniteScope(FileName, NumberOfRows, NumberOfColumns, h_ActualState);
			break;
		
		case UnitParameters:
			FileName        = "UnitParametersInSolverObject.txt";
			NumberOfRows    = ThreadConfiguration.TotalLogicalThreads;
			NumberOfColumns = NUP;
			WriteToFileUniteScope(FileName, NumberOfRows, NumberOfColumns, h_UnitParameters);
			break;
		
		case SystemParameters:
			FileName        = "SystemParametersInSolverObject.txt";
			NumberOfRows    = NS;
			NumberOfColumns = NSP;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_SystemParameters);
			break;
		
		case GlobalParameters:
			FileName        = "GlobalParametersInSolverObject.txt";
			NumberOfRows    = NGP;
			NumberOfColumns = 1;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_GlobalParameters);
			break;
		
		case IntegerGlobalParameters:
			FileName        = "IntegerGlobalParametersInSolverObject.txt";
			NumberOfRows    = NiGP;
			NumberOfColumns = 1;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_IntegerGlobalParameters);
			break;
		
		case UnitAccessories:
			FileName        = "UnitAccessoriesInSolverObject.txt";
			NumberOfRows    = ThreadConfiguration.TotalLogicalThreads;
			NumberOfColumns = NUA;
			WriteToFileUniteScope(FileName, NumberOfRows, NumberOfColumns, h_UnitAccessories);
			break;
		
		case IntegerUnitAccessories:
			FileName        = "IntegerUnitAccessoriesInSolverObject.txt";
			NumberOfRows    = ThreadConfiguration.TotalLogicalThreads;
			NumberOfColumns = NiUA;
			WriteToFileUniteScope(FileName, NumberOfRows, NumberOfColumns, h_IntegerUnitAccessories);
			break;
		
		case SystemAccessories:
			FileName        = "SystemAccessoriesInSolverObject.txt";
			NumberOfRows    = NS;
			NumberOfColumns = NSA;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_SystemAccessories);
			break;
		
		case IntegerSystemAccessories:
			FileName        = "IntegerSystemAccessoriesInSolverObject.txt";
			NumberOfRows    = NS;
			NumberOfColumns = NiSA;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_IntegerSystemAccessories);
			break;
		
		case CouplingStrength:
			FileName        = "CouplingStrengthInSolverObject.txt";
			NumberOfRows    = NS;
			NumberOfColumns = NC;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_CouplingStrength);
			break;
			
		case CouplingIndex:
			FileName        = "CouplingIndexInSolverObject.txt";
			NumberOfRows    = NC;
			NumberOfColumns = 1;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_CouplingIndex);
			break;
		
		default :
			std::cerr << "ERROR: In solver member function Print!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}	
}

// PRINT, Coupling matrix, dense output
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::Print(ListOfVariables Variable, int SerialNumber)
{
	std::string FileName;
	int NumberOfRows;
	int NumberOfColumns;
	
	switch (Variable)
	{
		case CouplingMatrix:
			BoundCheck("Print", "CouplingMatrix", SerialNumber, 0, NC-1);
			FileName = "CouplingMatrixInSolverObject_" + std::to_string(SerialNumber) + ".txt";
			NumberOfRows    = UPS;
			NumberOfColumns = UPS;
			WriteToFileSystemAndGlobalScope(FileName, NumberOfRows, NumberOfColumns, h_CouplingMatrixOriginal+SerialNumber*UPS*UPS);
			break;
		
		case DenseOutput:
			BoundCheck("Print", "DenseOutput", SerialNumber, 0, NS-1);
			FileName = "DenseOutput_" + std::to_string(SerialNumber)+ ".txt";
			NumberOfRows = h_DenseOutputIndex[SerialNumber];
			WriteToFileDenseOutput(FileName, NumberOfRows, SerialNumber, h_DenseOutputTimeInstances, h_DenseOutputStates);
			break;
		
		default :
			std::cerr << "ERROR: In solver member function Print!" << std::endl;
			std::cerr << "       Option: " << VariablesToString(Variable) << std::endl;
			std::cerr << "       This option needs different argument configuration or not applicable!" << std::endl;
			exit(EXIT_FAILURE);
	}
}

// SOLVE
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::Solve()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	//CoupledSystems_PerBlock_MultipleSystems_MultipleBlockLaunches<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision><<<ThreadConfiguration.GridSize, ThreadConfiguration.BlockSize, DynamicSharedMemoryRequired, Stream>>> (ThreadConfiguration, GlobalVariables, SharedMemoryUsage, SolverOptions);
	
	//CoupledSystems_PerBlock_SingleSystem_MultipleBlockLaunches<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision><<<ThreadConfiguration.GridSize, ThreadConfiguration.BlockSize, DynamicSharedMemoryRequired, Stream>>> (ThreadConfiguration, GlobalVariables, SharedMemoryUsage, SolverOptions);
	
	//CoupledSystems_PerBlock_MultipleSystems_SingleBlockLaunch<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision><<<ThreadConfiguration.GridSize, ThreadConfiguration.BlockSize, DynamicSharedMemoryRequired, Stream>>> (ThreadConfiguration, GlobalVariables, SharedMemoryUsage, SolverOptions);

	CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision><<<ThreadConfiguration.GridSize, ThreadConfiguration.BlockSize, DynamicSharedMemoryRequired, Stream>>> (ThreadConfiguration, GlobalVariables, SharedMemoryUsage, SolverOptions);
}

// SYNCHRONISE DEVICE
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SynchroniseDevice()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaDeviceSynchronize() );
}

// INSERT SYNCHRONISATION POINT
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::InsertSynchronisationPoint()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaEventRecord(Event, Stream) );
}

// SYNCHRONISE SOLVER
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SynchroniseSolver()
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

std::string SolverOptionsToString(ListOfSolverOptions Option)
{
	switch(Option)
	{
		case InitialTimeStep:
			return "InitialTimeStep";
		case ActiveSystems:
			return "ActiveSystems";
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
		case SharedGlobalVariables:
			return "SharedGlobalVariables";
		case SharedCouplingMatrices:
			return "SharedCouplingMatrices";
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
		case UnitParameters:
			return "UnitParameters";
		case SystemParameters:
			return "SystemParameters";
		case GlobalParameters:
			return "GlobalParameters";
		case IntegerGlobalParameters:
			return "IntegerGlobalParameters";
		case UnitAccessories:
			return "UnitAccessories";
		case IntegerUnitAccessories:
			return "IntegerUnitAccessories";
		case SystemAccessories:
			return "SystemAccessories";
		case IntegerSystemAccessories:
			return "IntegerSystemAccessories";
		case CouplingMatrix:
			return "CouplingMatrix";
		case CouplingStrength:
			return "CouplingStrength";
		case CouplingIndex:
			return "CouplingIndex";
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
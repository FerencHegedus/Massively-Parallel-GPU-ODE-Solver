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
enum ListOfSolverOptions{ InitialTimeStep, ActiveSystems, \
                          MaximumTimeStep, MinimumTimeStep, TimeStepGrowLimit, TimeStepShrinkLimit, MaxStepInsideEvent, MaximumNumberOfTimeSteps, \
						  RelativeTolerance, AbsoluteTolerance, \
						  EventTolerance, EventDirection, EventStopCounter, \
						  DenseOutputTimeStep };



void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);



template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
class ProblemSolver
{
    private:
		// Setup CUDA
		int Device;
		int Revision;
		cudaStream_t Stream;
		cudaEvent_t Event;
		
		// Thread management
		struct Struct_ThreadConfiguration
		{
			int LogicalThreadsPerBlock;
			int NumberOfBlockLaunches;
			int ThreadPadding;
			int BlockSize;
			int GridSize;
			int ThreadAllocationRequired;
		} ThreadConfiguration;
		
		// Global memory management
		size_t GlobalMemoryRequired;
		size_t GlobalMemoryFree;
		size_t GlobalMemoryTotal;
		
		long SizeOfTimeDomain;
		long SizeOfActualState;
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
		long SizeOfDenseOutputIndex;
		long SizeOfDenseOutputTimeInstances;
		long SizeOfDenseOutputStates;
		
		Precision* h_TimeDomain;
		Precision* h_ActualState;
		Precision* h_UnitParameters;
		Precision* h_SystemParameters;
		Precision* h_GlobalParameters;
		int*       h_IntegerGlobalParameters;
		Precision* h_UnitAccessories;
		int*       h_IntegerUnitAccessories;
		Precision* h_SystemAccessories;
		int*       h_IntegerSystemAccessories;
		Precision* h_CouplingMatrix;
		int*       h_DenseOutputIndex;
		Precision* h_DenseOutputTimeInstances;
		Precision* h_DenseOutputStates;
		
		struct Struct_GlobalVariables
		{
			Precision* d_TimeDomain;
			Precision* d_ActualState;
			Precision* d_UnitParameters;
			Precision* d_SystemParameters;
			Precision* d_GlobalParameters;
			int*       d_IntegerGlobalParameters;
			Precision* d_UnitAccessories;
			int*       d_IntegerUnitAccessories;
			Precision* d_SystemAccessories;
			int*       d_IntegerSystemAccessories;
			Precision* d_CouplingMatrix;
			int*       d_DenseOutputIndex;
			Precision* d_DenseOutputTimeInstances;
			Precision* d_DenseOutputStates;
			
			Precision* d_RelativeTolerance;
			Precision* d_AbsoluteTolerance;
			Precision* d_EventTolerance;
			int*       d_EventDirection;
			int*       d_EventStopCounter;
		} GlobalVariables;
		
		// Shared memory management
		struct Struct_SharedMemoryUsage
		{
			int GlobalVariables;     // Default: OFF
			int SystemVariables;     // Default: OFF
			int CouplingMatrices;    // Default: OFF
			int TolerancesAndEvents; // Default: OFF
			int CouplingTerms;       // Default: ON
		} SharedMemoryUsage;
		
		int SizeOfAlgorithmTolerances;
		
		size_t SharedMemoryRequired_GlobalVariables;    
		size_t SharedMemoryRequired_SystemVariables;   
		size_t SharedMemoryRequired_CouplingMatrices;    
		size_t SharedMemoryRequired_TolerancesAndEvents; 
		size_t SharedMemoryRequired_CouplingTerms;      
		size_t SharedMemoryRequired_UpperLimit;
		size_t SharedMemoryRequired_Actual;
		size_t SharedMemoryAvailable;
		
		// Constant memory management
		double h_BT_RKCK45[26];
		
		// Default solver options
		struct Struct_SolverOptions
		{
			Precision InitialTimeStep;
			int       ActiveSystems;
			Precision MaximumTimeStep;
			Precision MinimumTimeStep;
			Precision TimeStepGrowLimit;
			Precision TimeStepShrinkLimit;
			int       MaxStepInsideEvent;
			Precision DenseOutputTimeStep;
			int       MaximumNumberOfTimeSteps;
		} SolverOptions;
		
	public:
		ProblemSolver(int);
		~ProblemSolver();
		
		template <typename T>
		void SolverOption(ListOfSolverOptions, T);
		
		/*void SetHost(int, int, VariableSelection, int, double); // Unit scope
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
		void SynchroniseSolver();*/
};


// --- INCLUDE SOLVERS ---

#include "CoupledSystmes_PerBlock_RungeKutta.cuh"


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
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::ProblemSolver(int AssociatedDevice)
{
    std::cout << "---------------------------" << std::endl;
	std::cout << "Creating a SolverObject ..." << std::endl;
	std::cout << "---------------------------" << std::endl << std::endl;
	
	
	// Architecture specific setup
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
	
	
	// Thread management
	std::cout << "THREAD MANAGEMENT:" << std::endl;
	
	ThreadConfiguration.LogicalThreadsPerBlock   = SPB * UPS;
	ThreadConfiguration.NumberOfBlockLaunches    = ThreadConfiguration.LogicalThreadsPerBlock / TPB + (ThreadConfiguration.LogicalThreadsPerBlock % TPB == 0 ? 0:1);
	ThreadConfiguration.ThreadPadding            = ThreadConfiguration.NumberOfBlockLaunches * TPB - ThreadConfiguration.LogicalThreadsPerBlock;
	ThreadConfiguration.BlockSize                = TPB;
	ThreadConfiguration.GridSize                 = NS/SPB + (NS % SPB == 0 ? 0:1);
	ThreadConfiguration.ThreadAllocationRequired = (ThreadConfiguration.LogicalThreadsPerBlock + ThreadConfiguration.ThreadPadding) * ThreadConfiguration.GridSize;
	
	std::cout << "   Total number of systems:            " << NS << std::endl;
	std::cout << "   Systems per block:                  " << SPB << std::endl;
	std::cout << "   Logical threads per block required: " << ThreadConfiguration.LogicalThreadsPerBlock << std::endl;
	std::cout << "   GPU threads per block:              " << ThreadConfiguration.BlockSize << std::endl;
	std::cout << "   Number of block launches:           " << ThreadConfiguration.NumberOfBlockLaunches << std::endl;
	std::cout << "   Thread padding:                     " << ThreadConfiguration.ThreadPadding << std::endl;
	std::cout << "   GridSize (total number of blocks):  " << ThreadConfiguration.GridSize << std::endl;
	std::cout << "   Total logical threads required:     " << ThreadConfiguration.ThreadAllocationRequired << std::endl;
	std::cout << "   Thread efficinecy:                  " << (double)(NS*UPS)/ThreadConfiguration.ThreadAllocationRequired << std::endl;
	std::cout << "   Number of idle logical threads:     " << ThreadConfiguration.ThreadAllocationRequired - (NS*UPS) << std::endl << std::endl;
	
	
	// Global memory management
	std::cout << "GLOBAL MEMORY MANAGEMENT:" << std::endl;
	
	SizeOfTimeDomain               = (long) NS * 2;
	SizeOfActualState              = (long) ThreadConfiguration.ThreadAllocationRequired * UD;
	SizeOfUnitParameters           = (long) ThreadConfiguration.ThreadAllocationRequired * NUP;
	SizeOfSystemParameters         = (long) NS * NSP;
	SizeOfGlobalParameters         = (long) NGP;
	SizeOfIntegerGlobalParameters  = (long) NiGP;
	SizeOfUnitAccessories          = (long) ThreadConfiguration.ThreadAllocationRequired * NUA;
	SizeOfIntegerUnitAccessories   = (long) ThreadConfiguration.ThreadAllocationRequired * NiUA;
	SizeOfSystemAccessories        = (long) NS * NSA;
	SizeOfIntegerSystemAccessories = (long) NS * NiSA;
	SizeOfEvents                   = (long) ThreadConfiguration.ThreadAllocationRequired * NE;
	SizeOfCouplingMatrix           = (long) NC * UPS * UPS;
	SizeOfDenseOutputIndex         = (long) NS;
	SizeOfDenseOutputTimeInstances = (long) NS * NDO;
	SizeOfDenseOutputStates        = (long) UD * UPS * NS * NDO;
	
	GlobalMemoryRequired = sizeof(Precision) * ( SizeOfTimeDomain + SizeOfActualState + SizeOfUnitParameters + SizeOfSystemParameters + SizeOfGlobalParameters + SizeOfUnitAccessories + SizeOfSystemAccessories + SizeOfCouplingMatrix + SizeOfDenseOutputTimeInstances + SizeOfDenseOutputStates + 2*UD + NE) + \
						   sizeof(int) * ( SizeOfIntegerGlobalParameters + SizeOfIntegerUnitAccessories + SizeOfIntegerSystemAccessories + SizeOfDenseOutputIndex + 2*NE);
	
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
	h_UnitParameters           = AllocateHostPinnedMemory<Precision>( SizeOfUnitParameters );
	h_SystemParameters         = AllocateHostPinnedMemory<Precision>( SizeOfSystemParameters );
	h_GlobalParameters         = AllocateHostPinnedMemory<Precision>( SizeOfGlobalParameters );
	h_IntegerGlobalParameters  = AllocateHostPinnedMemory<int>( SizeOfIntegerGlobalParameters );
	h_UnitAccessories          = AllocateHostPinnedMemory<Precision>( SizeOfUnitAccessories );
	h_IntegerUnitAccessories   = AllocateHostPinnedMemory<int>( SizeOfIntegerUnitAccessories );
	h_SystemAccessories        = AllocateHostPinnedMemory<Precision>( SizeOfSystemAccessories );
	h_IntegerSystemAccessories = AllocateHostPinnedMemory<int>( SizeOfIntegerSystemAccessories );
	h_CouplingMatrix           = AllocateHostPinnedMemory<Precision>( SizeOfCouplingMatrix );
	h_DenseOutputIndex         = AllocateHostPinnedMemory<int>( SizeOfDenseOutputIndex );
	h_DenseOutputTimeInstances = AllocateHostPinnedMemory<Precision>( SizeOfDenseOutputTimeInstances );
	h_DenseOutputStates        = AllocateHostPinnedMemory<Precision>( SizeOfDenseOutputStates );
	
	GlobalVariables.d_TimeDomain               = AllocateDeviceMemory<Precision>( SizeOfTimeDomain );               // REGISTERS
	GlobalVariables.d_ActualState              = AllocateDeviceMemory<Precision>( SizeOfActualState );              // REGISTERS
	GlobalVariables.d_UnitParameters           = AllocateDeviceMemory<Precision>( SizeOfUnitParameters );           // REGISTERS
	GlobalVariables.d_SystemParameters         = AllocateDeviceMemory<Precision>( SizeOfSystemParameters );         // SHARED/GLOBAL
	GlobalVariables.d_GlobalParameters         = AllocateDeviceMemory<Precision>( SizeOfGlobalParameters );         // SHARED/GLOBAL
	GlobalVariables.d_IntegerGlobalParameters  = AllocateDeviceMemory<int>( SizeOfIntegerGlobalParameters );        // SHARED/GLOBAL
	GlobalVariables.d_UnitAccessories          = AllocateDeviceMemory<Precision>( SizeOfUnitAccessories );          // REGISTERS
	GlobalVariables.d_IntegerUnitAccessories   = AllocateDeviceMemory<int>( SizeOfIntegerUnitAccessories );         // REGISTERS
	GlobalVariables.d_SystemAccessories        = AllocateDeviceMemory<Precision>( SizeOfSystemAccessories );        // SHARED/GLOBAL
	GlobalVariables.d_IntegerSystemAccessories = AllocateDeviceMemory<int>( SizeOfIntegerSystemAccessories );       // SHARED/GLOBAL
	GlobalVariables.d_CouplingMatrix           = AllocateDeviceMemory<Precision>( SizeOfCouplingMatrix );           // SHARED/GLOBAL
	GlobalVariables.d_DenseOutputIndex         = AllocateDeviceMemory<int>( SizeOfDenseOutputIndex );               // REGISTERS
	GlobalVariables.d_DenseOutputTimeInstances = AllocateDeviceMemory<Precision>( SizeOfDenseOutputTimeInstances ); // GLOBAL
	GlobalVariables.d_DenseOutputStates        = AllocateDeviceMemory<Precision>( SizeOfDenseOutputStates );        // GLOBAL
	
	GlobalVariables.d_RelativeTolerance        = AllocateDeviceMemory<Precision>( UD );                             // SHARED/GLOBAL(ADAPTIVE)
	GlobalVariables.d_AbsoluteTolerance        = AllocateDeviceMemory<Precision>( UD );                             // SHARED/GLOBAL(ADAPTIVE)
	GlobalVariables.d_EventTolerance           = AllocateDeviceMemory<Precision>( NE );                             // SHARED/GLOBAL
	GlobalVariables.d_EventDirection           = AllocateDeviceMemory<int>( NE );                                   // SHARED/GLOBAL
	GlobalVariables.d_EventStopCounter         = AllocateDeviceMemory<int>( NE );                                   // SHARED/GLOBAL
	
	
	// Shared memory management
	std::cout << "SHARED MEMORY MANAGEMENT:" << std::endl;
	
	SharedMemoryUsage.GlobalVariables     = 0; // Default: OFF
	SharedMemoryUsage.SystemVariables     = 0; // Default: OFF
	SharedMemoryUsage.CouplingMatrices    = 0; // Default: OFF
	SharedMemoryUsage.TolerancesAndEvents = 0; // Default: OFF
	SharedMemoryUsage.CouplingTerms       = 1; // Default: ON
	
	switch (Algorithm)
	{
		case RK4:
			SizeOfAlgorithmTolerances = 0;
			break;
		default:
			SizeOfAlgorithmTolerances = 2*UD;
			break;
	}
	
	SharedMemoryRequired_GlobalVariables     = sizeof(Precision)*(SizeOfGlobalParameters) + sizeof(int)*(SizeOfIntegerGlobalParameters);
	SharedMemoryRequired_SystemVariables     = sizeof(Precision)*(SizeOfSystemParameters+SizeOfSystemAccessories) + sizeof(int)*(SizeOfIntegerSystemAccessories);
	SharedMemoryRequired_CouplingMatrices    = sizeof(Precision)*(SizeOfCouplingMatrix);
	SharedMemoryRequired_TolerancesAndEvents = sizeof(Precision)*(SizeOfAlgorithmTolerances+NE) + sizeof(int)*(2*NE);
	SharedMemoryRequired_CouplingTerms       = sizeof(Precision)*(NC*UPS);
	SharedMemoryRequired_UpperLimit          = SharedMemoryRequired_GlobalVariables + SharedMemoryRequired_SystemVariables + SharedMemoryRequired_CouplingMatrices + SharedMemoryRequired_TolerancesAndEvents + SharedMemoryRequired_CouplingTerms;
	
	SharedMemoryRequired_Actual = SharedMemoryUsage.GlobalVariables * SharedMemoryRequired_GlobalVariables + \
	                              SharedMemoryUsage.SystemVariables * SharedMemoryRequired_SystemVariables + \
								  SharedMemoryUsage.CouplingMatrices * SharedMemoryRequired_CouplingMatrices + \
								  SharedMemoryUsage.TolerancesAndEvents * SharedMemoryRequired_TolerancesAndEvents + \
								  SharedMemoryUsage.CouplingTerms * SharedMemoryRequired_CouplingTerms;
	
	SharedMemoryAvailable       = SelectedDeviceProperties.sharedMemPerBlock;
	
	std::cout << "   Required shared memory per block for different variables:" << std::endl;
	std::cout << "   Shared memory required by global variables:      " << SharedMemoryRequired_GlobalVariables << " b (OFF -> SolverOption)" << std::endl;
	std::cout << "   Shared memory required by system variables:      " << SharedMemoryRequired_SystemVariables << " b (OFF -> SolverOption)" << std::endl;
	std::cout << "   Shared memory required by coupling matrices:     " << SharedMemoryRequired_CouplingMatrices << " b (OFF -> SolverOption)" << std::endl;
	std::cout << "   Shared memory required by tolerances and events: " << SharedMemoryRequired_TolerancesAndEvents << " b (OFF -> SolverOption)" << std::endl;
	std::cout << "   Shared memory required by coupling terms:        " << SharedMemoryRequired_CouplingTerms << " b (ON -> SolverOption)" << std::endl << std::endl;
	
	std::cout << "   Upper limit of possible shared memory usage per block:  " << SharedMemoryRequired_UpperLimit << " b (All is ON)" << std::endl;
	std::cout << "   Actual shared memory required per block:                " << SharedMemoryRequired_Actual << " b" << std::endl;
	std::cout << "   Available shared memory per block:                      " << SharedMemoryAvailable << " b" << std::endl << std::endl;
	
	std::cout << "   Number of possible blocks per streaming multiprocessor: " << SharedMemoryAvailable/SharedMemoryRequired_Actual << std::endl;
	
	if ( SharedMemoryRequired_Actual >= SharedMemoryAvailable )
	{
        std::cout << std::endl;
		std::cout << "   WARNING: The required amount of shared memory is larger than the available!" << std::endl;
		std::cout << "            The solver kernel function cannot be run on the selected GPU!" << std::endl;
		std::cout << "            Turn OFF some variables using shared memory!" << std::endl;
    }
	std::cout << std::endl;
	
	
	// Constant memory management
	std::cout << "CONSTANT MEMORY MANAGEMENT:" << std::endl;
	
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
	
	gpuErrCHK( cudaMemcpyToSymbol(d_BT_RKCK45, h_BT_RKCK45, 26*sizeof(double)) );
	std::cout << std::endl;
	
	
	// Default values of Solver Options
	std::cout << "DEFAULT SOLVER OPTIONS:" << std::endl;
	
	SolverOptions.InitialTimeStep          = 1e-2;
	SolverOptions.ActiveSystems            = NS;
	SolverOptions.MaximumTimeStep          = 1.0e6;
	SolverOptions.MinimumTimeStep          = 1.0e-12;
	SolverOptions.TimeStepGrowLimit        = 5.0;
	SolverOptions.TimeStepShrinkLimit      = 0.1;
	SolverOptions.MaxStepInsideEvent       = 50;
	SolverOptions.DenseOutputTimeStep      = -1e-2;
	SolverOptions.MaximumNumberOfTimeSteps = 0;
	
	std::vector<Precision> DefaultAlgorithmTolerances(UD,1e-8);
	gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_RelativeTolerance, &DefaultAlgorithmTolerances, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
	gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_AbsoluteTolerance, &DefaultAlgorithmTolerances, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
	
	if ( NE > 0 )
	{
		std::vector<Precision> DefaultEventTolerance(NE,1e-6);
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventTolerance, &DefaultEventTolerance, sizeof(Precision), cudaMemcpyHostToDevice, Stream) );
		
		std::vector<int> EventStopCounterAndDirection(NE,0);
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventDirection,   &EventStopCounterAndDirection, sizeof(int), cudaMemcpyHostToDevice, Stream) );
		gpuErrCHK( cudaMemcpyAsync(GlobalVariables.d_EventStopCounter, &EventStopCounterAndDirection, sizeof(int), cudaMemcpyHostToDevice, Stream) );
	}
	
	std::cout << "   Initial time step:            " << SolverOptions.InitialTimeStep << std::endl;
	std::cout << "   Active systems:               " << SolverOptions.ActiveSystems << std::endl;
	std::cout << "   Maximum time step:            " << SolverOptions.MaximumTimeStep << std::endl;
	std::cout << "   Minimum time step:            " << SolverOptions.MinimumTimeStep << std::endl;
	std::cout << "   Time step grow limit:         " << SolverOptions.TimeStepGrowLimit << std::endl;
	std::cout << "   Time step shrink limit:       " << SolverOptions.TimeStepShrinkLimit << std::endl;
	std::cout << "   Max time step inside event:   " << SolverOptions.MaxStepInsideEvent << std::endl;
	std::cout << "   Dense output time step:       " << SolverOptions.DenseOutputTimeStep << std::endl;
	std::cout << "   Maximum number of time steps: " << SolverOptions.MaximumNumberOfTimeSteps << std::endl;
	std::cout << "   Algorithm absolute tolerance: " << 1e-8 << std::endl;
	std::cout << "   Algorithm relative tolerance: " << 1e-8 << std::endl;
	std::cout << "   Event absolute tolerance:     " << 1e-6 << std::endl;
	std::cout << "   Event stop counter:           " << 0 << std::endl;
	std::cout << "   Event direction of detection: " << 0 << std::endl;
	
	
	// End creating SolverObject
	std::cout << std::endl;
	
	std::cout << "Object for Parameters scan is successfully created!" << std::endl;
	std::cout << "Required memory allocations have been done" << std::endl;
	std::cout << "Coo man coo!!!" << std::endl << std::endl;
}

// DESTRUCTOR
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::~ProblemSolver()
{
    gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaStreamDestroy(Stream) );
	gpuErrCHK( cudaEventDestroy(Event) );
	
	gpuErrCHK( cudaFreeHost(h_TimeDomain) );
	gpuErrCHK( cudaFreeHost(h_ActualState) );
	gpuErrCHK( cudaFreeHost(h_UnitParameters) );
	gpuErrCHK( cudaFreeHost(h_SystemParameters) );
	gpuErrCHK( cudaFreeHost(h_GlobalParameters) );
	gpuErrCHK( cudaFreeHost(h_IntegerGlobalParameters) );
	gpuErrCHK( cudaFreeHost(h_UnitAccessories) );
	gpuErrCHK( cudaFreeHost(h_IntegerUnitAccessories) );
	gpuErrCHK( cudaFreeHost(h_SystemAccessories) );
	gpuErrCHK( cudaFreeHost(h_IntegerSystemAccessories) );
	gpuErrCHK( cudaFreeHost(h_CouplingMatrix) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputIndex) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFreeHost(h_DenseOutputStates) );
	
	gpuErrCHK( cudaFree(GlobalVariables.d_TimeDomain) );
	gpuErrCHK( cudaFree(GlobalVariables.d_ActualState) );
	gpuErrCHK( cudaFree(GlobalVariables.d_UnitParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_SystemParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_GlobalParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerGlobalParameters) );
	gpuErrCHK( cudaFree(GlobalVariables.d_UnitAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerUnitAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_SystemAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_IntegerSystemAccessories) );
	gpuErrCHK( cudaFree(GlobalVariables.d_CouplingMatrix) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputIndex) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputTimeInstances) );
	gpuErrCHK( cudaFree(GlobalVariables.d_DenseOutputStates) );
	gpuErrCHK( cudaFree(GlobalVariables.d_RelativeTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_AbsoluteTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventTolerance) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventDirection) );
	gpuErrCHK( cudaFree(GlobalVariables.d_EventStopCounter) );
	
	std::cout << "Object for Parameters scan is deleted!" << std::endl;
	std::cout << "Every memory have been deallocated!" << std::endl << std::endl;
}

// OPTION, int
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
template <typename T>
void ProblemSolver<NS,UPS,UD,TPB,SPB,NC,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,Algorithm,Precision>::SolverOption(ListOfSolverOptions Option, T Value)
{
	//std::cout << "Coo man coo:" << std::endl;
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
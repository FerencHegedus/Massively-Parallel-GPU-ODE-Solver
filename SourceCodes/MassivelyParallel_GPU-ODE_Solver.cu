#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>

#include "MassivelyParallel_GPU-ODE_Solver.cuh"

#define F(i)    F[tid + i*NT]
#define X(i)    X[tid + i*NT]
#define cPAR(i) cPAR[tid + i*NT]
#define sPAR(i) sPAR[i]
#define ACC(i)  ACC[tid + i*NT]
#define EF(i)   EF[tid + i*NT]
#define TD(i)   TD[tid + i*NT]

#include "PerThread_SystemDefinition.cuh"

#undef F
#undef X
#undef cPAR
#undef sPAR
#undef ACC
#undef EF
#undef TD

#include "PerThread_RungeKutta.cuh"

#define gpuErrCHK(call)                                                                \
{                                                                                      \
	const cudaError_t error = call;                                                    \
	if (error != cudaSuccess)                                                          \
	{                                                                                  \
		std::cout << "Error: " << __FILE__ << ":" << __LINE__ << std::endl;                      \
		std::cout << "code:" << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
		exit(1);                                                                       \
	}                                                                                  \
}

#define PI 3.14159265358979323846

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

ProblemSolver::ProblemSolver(const ConstructorConfiguration& Configuration, int AssociatedDevice)
{
    Device = AssociatedDevice;
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
	gpuErrCHK( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
	
	gpuErrCHK( cudaStreamCreate(&Stream) );
	gpuErrCHK( cudaEventCreate(&Event) );
	
	std::cout << "Creating a SolverObject ..." << std::endl;
	
	KernelParameters.NumberOfThreads = Configuration.NumberOfThreads;
	
	KernelParameters.SystemDimension           = Configuration.SystemDimension;
	KernelParameters.NumberOfControlParameters = Configuration.NumberOfControlParameters;
	KernelParameters.NumberOfSharedParameters  = Configuration.NumberOfSharedParameters;
	KernelParameters.NumberOfEvents            = Configuration.NumberOfEvents;
	KernelParameters.NumberOfAccessories       = Configuration.NumberOfAccessories;
	KernelParameters.DenseOutputNumberOfPoints = Configuration.DenseOutputNumberOfPoints;
	
	SizeOfTimeDomain        = KernelParameters.NumberOfThreads * 2;
	SizeOfActualState       = KernelParameters.NumberOfThreads * KernelParameters.SystemDimension;
	SizeOfControlParameters = KernelParameters.NumberOfThreads * KernelParameters.NumberOfControlParameters;
	SizeOfSharedParameters  = KernelParameters.NumberOfSharedParameters;
	SizeOfAccessories       = KernelParameters.NumberOfThreads * KernelParameters.NumberOfAccessories;
	SizeOfEvents            = KernelParameters.NumberOfThreads * KernelParameters.NumberOfEvents;
	
	SizeOfDenseOutputIndex         = KernelParameters.NumberOfThreads;
	SizeOfDenseOutputTimeInstances = KernelParameters.NumberOfThreads * KernelParameters.DenseOutputNumberOfPoints;
	SizeOfDenseOutputStates        = KernelParameters.NumberOfThreads * KernelParameters.SystemDimension * KernelParameters.DenseOutputNumberOfPoints;
	
	GlobalMemoryRequired = sizeof(double) * ( SizeOfTimeDomain + 11*SizeOfActualState + SizeOfControlParameters + SizeOfSharedParameters + SizeOfAccessories + \
                                              2*SizeOfEvents + 2*KernelParameters.SystemDimension + KernelParameters.NumberOfEvents + \
											  SizeOfDenseOutputTimeInstances + SizeOfDenseOutputStates ) + \
						   sizeof(int) * ( 2*SizeOfEvents + 2*KernelParameters.NumberOfEvents + SizeOfDenseOutputIndex );
	
	cudaMemGetInfo( &GlobalMemoryFree, &GlobalMemoryTotal );
	std::cout << "   Required global memory:       " << GlobalMemoryRequired/1024/1024 << "Mb" << std::endl;
	std::cout << "   Available free global memory: " << GlobalMemoryFree/1024/1024     << "Mb" << std::endl;
	std::cout << "   Total global memory:          " << GlobalMemoryTotal/1024/1024    << "Mb" << std::endl;
	
	if ( GlobalMemoryRequired >= GlobalMemoryFree )
	{
        std::cerr << "   ERROR: the required amount of global memory is larger than the free!" << std::endl;
		std::cerr << "          Try to reduce the number of points of the DenseOutput or reduce the NumberOfThreads!" << std::endl;
        exit(EXIT_FAILURE);
    }
	
	std::cout << std::endl;
	
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
	
	DynamicSharedMemoryRKCK45     = 2*KernelParameters.SystemDimension*sizeof(double) + KernelParameters.NumberOfEvents*( sizeof(int) + sizeof(double) + sizeof(int) ) + KernelParameters.NumberOfSharedParameters*sizeof(double);
	DynamicSharedMemoryRKCK45_EH0 = 2*KernelParameters.SystemDimension*sizeof(double) + KernelParameters.NumberOfSharedParameters*sizeof(double);
	DynamicSharedMemoryRK4        = KernelParameters.NumberOfEvents*( sizeof(int) + sizeof(double) + sizeof(int) ) + KernelParameters.NumberOfSharedParameters*sizeof(double);
	DynamicSharedMemoryRK4_EH0    = KernelParameters.NumberOfSharedParameters*sizeof(double);
	
	
	h_TimeDomain        = AllocateHostPinnedMemory<double>( SizeOfTimeDomain );
	h_ActualState       = AllocateHostPinnedMemory<double>( SizeOfActualState );
	h_ControlParameters = AllocateHostPinnedMemory<double>( SizeOfControlParameters );
	h_SharedParameters  = AllocateHostPinnedMemory<double>( SizeOfSharedParameters );
	h_Accessories       = AllocateHostPinnedMemory<double>( SizeOfAccessories );
	
	h_DenseOutputIndex         = AllocateHostPinnedMemory<int>( SizeOfDenseOutputIndex );
	h_DenseOutputTimeInstances = AllocateHostPinnedMemory<double>( SizeOfDenseOutputTimeInstances );
	h_DenseOutputStates        = AllocateHostPinnedMemory<double>( SizeOfDenseOutputStates );
	
	
	KernelParameters.d_TimeDomain        = AllocateDeviceMemory<double>( SizeOfTimeDomain );
	KernelParameters.d_ActualState       = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_ControlParameters = AllocateDeviceMemory<double>( SizeOfControlParameters );
	KernelParameters.d_SharedParameters  = AllocateDeviceMemory<double>( SizeOfSharedParameters );
	KernelParameters.d_Accessories       = AllocateDeviceMemory<double>( SizeOfAccessories );
	
	KernelParameters.d_State    = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_Stages   = AllocateDeviceMemory<double>( SizeOfActualState * 6 );
	
	KernelParameters.d_NextState = AllocateDeviceMemory<double>( SizeOfActualState );
	
	KernelParameters.d_Error           = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_ActualTolerance = AllocateDeviceMemory<double>( SizeOfActualState );
	
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
	
	SolverType = RKCK45;
	
	KernelParameters.MaximumTimeStep     = 1.0e6;
	KernelParameters.MinimumTimeStep     = 1.0e-12;
	KernelParameters.TimeStepGrowLimit   = 5.0;
	KernelParameters.TimeStepShrinkLimit = 0.1;
	
	KernelParameters.MaxStepInsideEvent  = 50;
	
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, AssociatedDevice);
	BlockSize  = SelectedDeviceProperties.warpSize;
	
	GridSize = KernelParameters.NumberOfThreads/BlockSize + (KernelParameters.NumberOfThreads % BlockSize == 0 ? 0:1);
	
	KernelParameters.DenseOutputEnabled  = 0;
	KernelParameters.DenseOutputTimeStep = -1e-2;
	
	KernelParameters.MaximumNumberOfTimeSteps = 0;
	
	std::cout << "   Total shared memory required:  " << DynamicSharedMemoryRKCK45 / 1024                  << " Kb" << std::endl;
	std::cout << "   Total shared memory available: " << SelectedDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
	
	if ( DynamicSharedMemoryRKCK45 >= SelectedDeviceProperties.sharedMemPerBlock )
	{
        std::cerr << "   ERROR: the required amount of shared memory is larger than the free!" << std::endl;
		std::cerr << "          Try to reduce the number the SharedParameters!" << std::endl;
        exit(EXIT_FAILURE);
    }
	
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

ProblemSolver::~ProblemSolver()
{
    gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaStreamDestroy(Stream) );
	gpuErrCHK( cudaEventDestroy(Event) );
	
	gpuErrCHK( cudaFreeHost(h_TimeDomain) );
	gpuErrCHK( cudaFreeHost(h_ActualState) );
	gpuErrCHK( cudaFreeHost(h_ControlParameters) );
	gpuErrCHK( cudaFreeHost(h_SharedParameters) );
	gpuErrCHK( cudaFreeHost(h_Accessories) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_TimeDomain) );
	gpuErrCHK( cudaFree(KernelParameters.d_ActualState) );
	gpuErrCHK( cudaFree(KernelParameters.d_ControlParameters) );
	gpuErrCHK( cudaFree(KernelParameters.d_SharedParameters) );
	gpuErrCHK( cudaFree(KernelParameters.d_Accessories) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_State) );
	gpuErrCHK( cudaFree(KernelParameters.d_Stages) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_NextState) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_Error) );
	gpuErrCHK( cudaFree(KernelParameters.d_ActualTolerance) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_ActualEventValue) );
	gpuErrCHK( cudaFree(KernelParameters.d_NextEventValue) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventCounter) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventEquilibriumCounter) );
	
	gpuErrCHK( cudaFree(KernelParameters.d_RelativeTolerance) );
	gpuErrCHK( cudaFree(KernelParameters.d_AbsoluteTolerance) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventTolerance) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventDirection) );
	gpuErrCHK( cudaFree(KernelParameters.d_EventStopCounter) );
	
	std::cout << "Object for Parameters scan is deleted! Every memory have been deallocated!" << std::endl << std::endl;
}

// Problem scope
void ProblemSolver::SetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber, double Value)
{
	if ( ProblemNumber >= KernelParameters.NumberOfThreads )
	{
        std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
		     << "The index of the problem number cannot be larger than " << KernelParameters.NumberOfThreads-1 << "! "\
			 << "(The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	switch (Variable)
	{
		case TimeDomain:
			if ( SerialNumber >= 2 )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the TimeDomain cannot be larger than " << 2-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_TimeDomain[idx] = Value;
			break;
		
		case ActualState:
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the ActualState cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_ActualState[idx] = Value;
			break;
		
		case ControlParameters:
			if ( SerialNumber >= KernelParameters.NumberOfControlParameters )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the ControlParameters cannot be larger than " << KernelParameters.NumberOfControlParameters-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_ControlParameters[idx] = Value;
			break;
		
		case Accessories:
			if ( SerialNumber >= KernelParameters.NumberOfAccessories )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the Accessories cannot be larger than " << KernelParameters.NumberOfAccessories-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_Accessories[idx] = Value;
			break;
		
		case DenseTime:
			if ( SerialNumber >= KernelParameters.DenseOutputNumberOfPoints )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the DenseTime cannot be larger than " << KernelParameters.DenseOutputNumberOfPoints-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_DenseOutputTimeInstances[idx] = Value;
			break;
		
		default :
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Dense state
void ProblemSolver::SetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber, int TimeStep, double Value)
{
	if ( ProblemNumber >= KernelParameters.NumberOfThreads )
	{
        std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
		     << "The index of the problem number cannot be larger than " << KernelParameters.NumberOfThreads-1 << "! "\
			 << "(The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads + TimeStep*KernelParameters.NumberOfThreads*KernelParameters.SystemDimension;
	
	switch (Variable)
	{
		case DenseState:
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the DenseState cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			if ( TimeStep >= KernelParameters.DenseOutputNumberOfPoints )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The time step number of the DenseState cannot be larger than " << KernelParameters.DenseOutputNumberOfPoints-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_DenseOutputStates[idx] = Value;
			break;
		
		default :
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Global scope
void ProblemSolver::SetHost(VariableSelection Variable, int SerialNumber, double Value)
{
	switch (Variable)
	{
		case SharedParameters:
			if ( SerialNumber >= KernelParameters.NumberOfSharedParameters )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the SharedParameters cannot be larger than " << KernelParameters.NumberOfSharedParameters-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_SharedParameters[SerialNumber] = Value;
			break;
		
		default :
			std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

void ProblemSolver::SynchroniseFromHostToDevice(VariableSelection Variable)
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
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputIndex,                 h_DenseOutputIndex,            SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputTimeInstances, h_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputStates,               h_DenseOutputStates,        SizeOfDenseOutputStates*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
			
		case All:
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_TimeDomain,               h_TimeDomain,        SizeOfTimeDomain*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ActualState,             h_ActualState,       SizeOfActualState*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ControlParameters, h_ControlParameters, SizeOfControlParameters*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_SharedParameters,   h_SharedParameters,  SizeOfSharedParameters*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_Accessories,             h_Accessories,       SizeOfAccessories*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputIndex,                 h_DenseOutputIndex,            SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputTimeInstances, h_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_DenseOutputStates,               h_DenseOutputStates,        SizeOfDenseOutputStates*sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		default :
			std::cerr << "ERROR in solver member function SynchroniseFromHostToDevice:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

void ProblemSolver::SynchroniseFromDeviceToHost(VariableSelection Variable)
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
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex,                 KernelParameters.d_DenseOutputIndex,            SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, KernelParameters.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates,               KernelParameters.d_DenseOutputStates,        SizeOfDenseOutputStates*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
			
		case All:
			gpuErrCHK( cudaMemcpyAsync(h_TimeDomain,               KernelParameters.d_TimeDomain,        SizeOfTimeDomain*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ActualState,             KernelParameters.d_ActualState,       SizeOfActualState*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, KernelParameters.d_ControlParameters, SizeOfControlParameters*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_SharedParameters,   KernelParameters.d_SharedParameters,  SizeOfSharedParameters*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_Accessories,             KernelParameters.d_Accessories,       SizeOfAccessories*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputIndex,                 KernelParameters.d_DenseOutputIndex,            SizeOfDenseOutputIndex*sizeof(int), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputTimeInstances, KernelParameters.d_DenseOutputTimeInstances, SizeOfDenseOutputTimeInstances*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			gpuErrCHK( cudaMemcpyAsync(h_DenseOutputStates,               KernelParameters.d_DenseOutputStates,        SizeOfDenseOutputStates*sizeof(double), cudaMemcpyDeviceToHost, Stream) );
			break;
		
		default :
			std::cerr << "ERROR in solver member function SynchroniseFromDeviceToHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Problem scope
double ProblemSolver::GetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber)
{
	if ( ProblemNumber >= KernelParameters.NumberOfThreads )
	{
        std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
		     << "The index of the problem number cannot be larger than " << KernelParameters.NumberOfThreads-1 << "! "\
			 << "(The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	double Value;
	switch (Variable)
	{
		case TimeDomain:
			if ( SerialNumber >= 2 )
			{
				std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
				     << "The serial number of the TimeDomain cannot be larger than " << 2-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_TimeDomain[idx];
			break;
			
		case ActualState:
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
				     << "The serial number of the ActualState cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_ActualState[idx];
			break;
			
		case ControlParameters:
			if ( SerialNumber >= KernelParameters.NumberOfControlParameters )
			{
				std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
				     << "The serial number of the ControlParameters cannot be larger than " << KernelParameters.NumberOfControlParameters-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_ControlParameters[idx];
			break;
			
		case Accessories:
			if ( SerialNumber >= KernelParameters.NumberOfAccessories )
			{
				std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
				     << "The serial number of the Accessories cannot be larger than " << KernelParameters.NumberOfAccessories-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_Accessories[idx];
			break;
		
		case DenseTime:
			if ( SerialNumber >= KernelParameters.DenseOutputNumberOfPoints )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the DenseTime cannot be larger than " << KernelParameters.DenseOutputNumberOfPoints-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_DenseOutputTimeInstances[idx];
			break;
		
		default :
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// Dense state
double ProblemSolver::GetHost(int ProblemNumber, VariableSelection Variable, int SerialNumber, int TimeStep)
{
	if ( ProblemNumber >= KernelParameters.NumberOfThreads )
	{
        std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
		     << "The index of the problem number cannot be larger than " << KernelParameters.NumberOfThreads-1 << "! "\
			 << "(The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads + TimeStep*KernelParameters.NumberOfThreads*KernelParameters.SystemDimension;
	
	double Value;
	switch (Variable)
	{
		case DenseState:
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The serial number of the DenseState cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			if ( TimeStep >= KernelParameters.DenseOutputNumberOfPoints )
			{
				std::cerr << "ERROR in solver member function SetHost:" << std::endl << "    "\
				     << "The time step number of the DenseState cannot be larger than " << KernelParameters.DenseOutputNumberOfPoints-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_DenseOutputStates[idx];
			break;
		
		default :
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// Global scope
double ProblemSolver::GetHost(VariableSelection Variable, int SerialNumber)
{
	double Value;
	switch (Variable)
	{
		case SharedParameters:
			if ( SerialNumber >= KernelParameters.NumberOfSharedParameters )
			{
				std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
				     << "The serial number of the SharedParameters cannot be larger than " << KernelParameters.NumberOfSharedParameters-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_SharedParameters[SerialNumber];
			break;
		
		default :
			std::cerr << "ERROR in solver member function GetHost:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

void ProblemSolver::Print(VariableSelection Variable)
{
	std::ofstream DataFile;
	int NumberOfRows;
	int NumberOfColumns;
	double* PointerToActualData;
	
	switch (Variable)
	{
		case TimeDomain:		DataFile.open ( "TimeDomainInSolverObject.txt" );
								NumberOfRows = KernelParameters.NumberOfThreads;
								NumberOfColumns = 2;
								PointerToActualData = h_TimeDomain; break;
		case ActualState:		DataFile.open ( "ActualStateInSolverObject.txt" );
								NumberOfRows = KernelParameters.NumberOfThreads;
								NumberOfColumns = KernelParameters.SystemDimension;
								PointerToActualData = h_ActualState; break;
		case ControlParameters: DataFile.open ( "ControlParametersInSolverObject.txt" );
								NumberOfRows = KernelParameters.NumberOfThreads;
								NumberOfColumns = KernelParameters.NumberOfControlParameters;
								PointerToActualData = h_ControlParameters; break;
		case SharedParameters:  DataFile.open ( "SharedParametersInSolverObject.txt" );
								NumberOfRows    = KernelParameters.NumberOfSharedParameters;		
								NumberOfColumns = 1;
								PointerToActualData = h_SharedParameters; break;				  
		case Accessories:		DataFile.open ( "AccessoriesInSolverObject.txt" );
								NumberOfRows = KernelParameters.NumberOfThreads;
								NumberOfColumns = KernelParameters.NumberOfAccessories;
								PointerToActualData = h_Accessories; break;
		
		default :
			std::cerr << "ERROR in solver member function Print:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
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

void ProblemSolver::Print(VariableSelection Variable, int ThreadID)
{
	if ( ( ThreadID >= KernelParameters.NumberOfThreads ) || ( ThreadID < 0 ) )
	{
        std::cerr << "ERROR in solver member function Print:" << std::endl << "    "\
		     << "The index of the problem number cannot be larger than " << KernelParameters.NumberOfThreads-1 << "! "\
			 << "(The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( Variable != DenseOutput )
	{
        std::cerr << "ERROR in solver member function Print:" << std::endl << "    "\
			 << "Invalid option for variable selection!\n";
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
	
	DataFile << "Accessories:\n";
	for (int i=0; i<KernelParameters.NumberOfAccessories; i++)
	{
		idx = ThreadID + i*KernelParameters.NumberOfThreads;
		DataFile.width(Width); DataFile << h_Accessories[idx];
		if ( i<(KernelParameters.NumberOfAccessories-1) )
			DataFile << ',';
	}
	DataFile << "\n\n";
	
	DataFile << "Time series:\n";
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
	
	DataFile.close();
}

// int type
void ProblemSolver::SolverOption(ListOfSolverOptions Option, int Value)
{
	switch (Option)
	{
		// Valid int type options
		case ThreadsPerBlock:
			BlockSize = Value;
			GridSize  = KernelParameters.NumberOfThreads/BlockSize + (KernelParameters.NumberOfThreads % BlockSize == 0 ? 0:1);
			break;
		
		case ActiveNumberOfThreads:
			KernelParameters.ActiveThreads = Value;
			break;
		
		case MaxStepInsideEvent:
			KernelParameters.MaxStepInsideEvent = Value;
			break;
		
		case DenseOutputEnabled:
			KernelParameters.DenseOutputEnabled = Value;
			break;
		
		case MaximumNumberOfTimeSteps:
			KernelParameters.MaximumNumberOfTimeSteps = Value;
			break;
		
		// Invalid option types
		case InitialTimeStep:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for InitialTimeStep is double! AUTOMATIC CONVERSION TO DOUBLE: " << (double)Value << std::endl << std::endl;
			KernelParameters.InitialTimeStep = (double)Value;
			break;
		
		case MaximumTimeStep:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for MaximumTimeStep is double! AUTOMATIC CONVERSION TO DOUBLE: " << (double)Value << std::endl << std::endl;
			KernelParameters.MaximumTimeStep = (double)Value;
			break;
		
		case MinimumTimeStep:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for MinimumTimeStep is double! AUTOMATIC CONVERSION TO DOUBLE: " << (double)Value << std::endl << std::endl;
			KernelParameters.MinimumTimeStep = (double)Value;
			break;
		
		case TimeStepGrowLimit:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for TimeStepGrowLimit is double! AUTOMATIC CONVERSION TO DOUBLE: " << (double)Value << std::endl << std::endl;
			KernelParameters.TimeStepGrowLimit = (double)Value;
			break;
		
		case TimeStepShrinkLimit:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for TimeStepShrinkLimit is double! AUTOMATIC CONVERSION TO DOUBLE: " << (double)Value << std::endl << std::endl;
			KernelParameters.TimeStepShrinkLimit = (double)Value;
			break;
		
		case DenseOutputTimeStep:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for DenseOutputTimeStep is double! AUTOMATIC CONVERSION TO DOUBLE: " << (double)Value << std::endl << std::endl;
			KernelParameters.DenseOutputTimeStep = (double)Value;
			break;
		
		default :
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// double type
void ProblemSolver::SolverOption(ListOfSolverOptions Option, double Value)
{
	switch (Option)
	{		
		// Valid double type options
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
		
		// Invalid option types
		case ThreadsPerBlock:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for ThreadsPerBlock is int! AUTOMATIC CONVERSION TO INT: " << (int)Value << std::endl << std::endl;
			BlockSize = (int)Value;
			GridSize  = KernelParameters.NumberOfThreads/BlockSize + (KernelParameters.NumberOfThreads % BlockSize == 0 ? 0:1);
			break;
		
		case ActiveNumberOfThreads:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for ActiveNumberOfThreads is int! AUTOMATIC CONVERSION TO INT: " << (int)Value << std::endl << std::endl;
			KernelParameters.ActiveThreads = (int)Value;
			break;
		
		case MaxStepInsideEvent:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for MaxStepInsideEvent is int! AUTOMATIC CONVERSION TO INT: " << (int)Value << std::endl << std::endl;
			KernelParameters.MaxStepInsideEvent = (int)Value;
			break;
		
		case DenseOutputEnabled:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for DenseOutputEnabled is int! AUTOMATIC CONVERSION TO INT: " << (int)Value << std::endl << std::endl;
			KernelParameters.DenseOutputEnabled = (int)Value;
			break;
		
		case MaximumNumberOfTimeSteps:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for MaximumNumberOfTimeSteps is int! AUTOMATIC CONVERSION TO INT: " << (int)Value << std::endl << std::endl;
			KernelParameters.MaximumNumberOfTimeSteps = (int)Value;
			break;
		
		default :
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Array of int
void ProblemSolver::SolverOption(ListOfSolverOptions Option, int SerialNumber, int Value)
{
	double ConvertedToDouble = (double)Value;
	
	switch (Option)
	{		
		// Valid int type options
		case EventDirection:
			if ( SerialNumber >= KernelParameters.NumberOfEvents )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the EventDirection cannot be larger than " << KernelParameters.NumberOfEvents-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventDirection+SerialNumber, &Value, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case EventStopCounter:
			if ( SerialNumber >= KernelParameters.NumberOfEvents )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the EventStopCounter cannot be larger than " << KernelParameters.NumberOfEvents-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventStopCounter+SerialNumber, &Value, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		// Invalid option types
		case RelativeTolerance:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for RelativeTolerance is double! AUTOMATIC CONVERSION TO DOUBLE: " << ConvertedToDouble << std::endl << std::endl;
			
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the RelativeTolerance cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_RelativeTolerance+SerialNumber, &ConvertedToDouble, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case AbsoluteTolerance:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for AbsoluteTolerance is double! AUTOMATIC CONVERSION TO DOUBLE: " << ConvertedToDouble << std::endl << std::endl;
			
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the AbsoluteTolerance cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_AbsoluteTolerance+SerialNumber, &ConvertedToDouble, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case EventTolerance:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for EventTolerance is double! AUTOMATIC CONVERSION TO DOUBLE: " << ConvertedToDouble << std::endl << std::endl;
			
			if ( SerialNumber >= KernelParameters.NumberOfEvents )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the EventTolerance cannot be larger than " << KernelParameters.NumberOfEvents-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventTolerance+SerialNumber, &ConvertedToDouble, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		default :
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Array of double
void ProblemSolver::SolverOption(ListOfSolverOptions Option, int SerialNumber, double Value)
{
	int ConvertedToInt = (int)Value;
	
	switch (Option)
	{		
		// Valid double type options
		case RelativeTolerance:
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the RelativeTolerance cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_RelativeTolerance+SerialNumber, &Value, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case AbsoluteTolerance:
			if ( SerialNumber >= KernelParameters.SystemDimension )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the AbsoluteTolerance cannot be larger than " << KernelParameters.SystemDimension-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_AbsoluteTolerance+SerialNumber, &Value, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case EventTolerance:
			if ( SerialNumber >= KernelParameters.NumberOfEvents )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the EventTolerance cannot be larger than " << KernelParameters.NumberOfEvents-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventTolerance+SerialNumber, &Value, sizeof(double), cudaMemcpyHostToDevice, Stream) );
			break;
		
		// Invalid option types
		case EventDirection:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for EventDirection is int! AUTOMATIC CONVERSION TO INT: " << ConvertedToInt << std::endl << std::endl;
			
			if ( SerialNumber >= KernelParameters.NumberOfEvents )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the EventDirection cannot be larger than " << KernelParameters.NumberOfEvents-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventDirection+SerialNumber, &ConvertedToInt, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		case EventStopCounter:
			std::cerr << "WARNING in solver member function SolverOption:" << std::endl << "    "\
			     << "Expected data type for EventStopCounter is int! AUTOMATIC CONVERSION TO INT: " << ConvertedToInt << std::endl << std::endl;
			
			if ( SerialNumber >= KernelParameters.NumberOfEvents )
			{
				std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
				     << "The serial number of the EventStopCounter cannot be larger than " << KernelParameters.NumberOfEvents-1 << "! "\
					 << "(The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			
			gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_EventStopCounter+SerialNumber, &ConvertedToInt, sizeof(int), cudaMemcpyHostToDevice, Stream) );
			break;
		
		default :
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// ListOfSolverAlgorithms type
void ProblemSolver::SolverOption(ListOfSolverOptions Option, ListOfSolverAlgorithms Value)
{
	switch (Option)
	{		
		case Solver:
			SolverType = Value;
			break;
			
		default :
			std::cerr << "ERROR in solver member function SolverOption:" << std::endl << "    "\
			     << "Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

void ProblemSolver::Solve()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	if ( SolverType==RKCK45 )
		PerThread_RKCK45<<<GridSize, BlockSize, DynamicSharedMemoryRKCK45, Stream>>> (KernelParameters);
	
	if ( SolverType==RKCK45_EH0 )
		PerThread_RKCK45_EH0<<<GridSize, BlockSize, DynamicSharedMemoryRKCK45_EH0, Stream>>> (KernelParameters);
	
	if ( SolverType==RK4 )
		PerThread_RK4<<<GridSize, BlockSize, DynamicSharedMemoryRK4, Stream>>> (KernelParameters);
	
	if ( SolverType==RK4_EH0 )
		PerThread_RK4_EH0<<<GridSize, BlockSize, DynamicSharedMemoryRK4_EH0, Stream>>> (KernelParameters);
}

void ProblemSolver::SynchroniseDevice()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaDeviceSynchronize() );
}

void ProblemSolver::InsertSynchronisationPoint()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaEventRecord(Event, Stream) );
}

void ProblemSolver::SynchroniseSolver()
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
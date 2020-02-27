#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>

#include "MassivelyParallel_GPU-ODE_Solver_PerBlockCoupling.cuh"

#include "PerThread_SystemDefinition_PerBlockCoupling.cuh"
#include "PerThread_RungeKutta_PerBlockCoupling.cuh"

#define gpuErrCHK(call)                                                                \
{                                                                                      \
	const cudaError_t error = call;                                                    \
	if (error != cudaSuccess)                                                          \
	{                                                                                  \
		cout << "Error: " << __FILE__ << ":" << __LINE__ << endl;                      \
		cout << "code:" << error << ", reason: " << cudaGetErrorString(error) << endl; \
		exit(1);                                                                       \
	}                                                                                  \
}

//#define PI 3.14159265358979323846

// --- CUDA DEVICE FUNCTIONS ---

void ListCUDADevices()
{
	int NumberOfDevices;
	cudaGetDeviceCount(&NumberOfDevices);
	for (int i = 0; i < NumberOfDevices; i++)
	{
		cudaDeviceProp CurrentDeviceProperties;
		cudaGetDeviceProperties(&CurrentDeviceProperties, i);
		cout << endl;
		cout << "Device number: " << i << endl;
		cout << "Device name:   " << CurrentDeviceProperties.name << endl;
		cout << "--------------------------------------" << endl;
		cout << "Total global memory:        " << CurrentDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << endl;
		cout << "Total shared memory:        " << CurrentDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << endl;
		cout << "Number of 32-bit registers: " << CurrentDeviceProperties.regsPerBlock << endl;
		cout << "Total constant memory:      " << CurrentDeviceProperties.totalConstMem / 1024 << " Kb" << endl;
		cout << endl;
		cout << "Number of multiprocessors:  " << CurrentDeviceProperties.multiProcessorCount << endl;
		cout << "Compute capability:         " << CurrentDeviceProperties.major << "." << CurrentDeviceProperties.minor << endl;
		cout << "Core clock rate:            " << CurrentDeviceProperties.clockRate / 1000 << " MHz" << endl;
		cout << "Memory clock rate:          " << CurrentDeviceProperties.memoryClockRate / 1000 << " MHz" << endl;
		cout << "Memory bus width:           " << CurrentDeviceProperties.memoryBusWidth  << " bits" << endl;
		cout << "Peak memory bandwidth:      " << 2.0*CurrentDeviceProperties.memoryClockRate*(CurrentDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << endl;
		cout << endl;
		cout << "Warp size:                  " << CurrentDeviceProperties.warpSize << endl;
		cout << "Max. warps per multiproc:   " << CurrentDeviceProperties.maxThreadsPerMultiProcessor / CurrentDeviceProperties.warpSize << endl;
		cout << "Max. threads per multiproc: " << CurrentDeviceProperties.maxThreadsPerMultiProcessor << endl;
		cout << "Max. threads per block:     " << CurrentDeviceProperties.maxThreadsPerBlock << endl;
		cout << "Max. block dimensions:      " << CurrentDeviceProperties.maxThreadsDim[0] << " * " << CurrentDeviceProperties.maxThreadsDim[1] << " * " << CurrentDeviceProperties.maxThreadsDim[2] << endl;
		cout << "Max. grid dimensions:       " << CurrentDeviceProperties.maxGridSize[0] << " * " << CurrentDeviceProperties.maxGridSize[1] << " * " << CurrentDeviceProperties.maxGridSize[2] << endl;
		cout << endl;
		cout << "Concurrent memory copy:     " << CurrentDeviceProperties.deviceOverlap << endl;
		cout << "Execution multiple kernels: " << CurrentDeviceProperties.concurrentKernels << endl;
		cout << "ECC support turned on:      " << CurrentDeviceProperties.ECCEnabled << endl << endl;
	}
	cout << endl;
}

int SelectDeviceByClosestRevision(int MajorRevision, int MinorRevision)
{
	int SelectedDevice;
	
	cudaDeviceProp SelectedDeviceProperties;
	memset( &SelectedDeviceProperties, 0, sizeof(cudaDeviceProp) );
	
	SelectedDeviceProperties.major = MajorRevision;
	SelectedDeviceProperties.minor = MinorRevision;
		cudaChooseDevice( &SelectedDevice, &SelectedDeviceProperties );
	
	cout << "CUDA Device Number Closest to Revision " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << ": " << SelectedDevice << endl << endl << endl;
	
	return SelectedDevice;
}

void PrintPropertiesOfSpecificDevice(int SelectedDevice)
{
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, SelectedDevice);
	
	cout << "Selected device number: " << SelectedDevice << endl;
	cout << "Selected device name:   " << SelectedDeviceProperties.name << endl;
	cout << "--------------------------------------" << endl;
	cout << "Total global memory:        " << SelectedDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << endl;
	cout << "Total shared memory:        " << SelectedDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << endl;
	cout << "Number of 32-bit registers: " << SelectedDeviceProperties.regsPerBlock << endl;
	cout << "Total constant memory:      " << SelectedDeviceProperties.totalConstMem / 1024 << " Kb" << endl;
	cout << endl;
	cout << "Number of multiprocessors:  " << SelectedDeviceProperties.multiProcessorCount << endl;
	cout << "Compute capability:         " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << endl;
	cout << "Core clock rate:            " << SelectedDeviceProperties.clockRate / 1000 << " MHz" << endl;
	cout << "Memory clock rate:          " << SelectedDeviceProperties.memoryClockRate / 1000 << " MHz" << endl;
	cout << "Memory bus width:           " << SelectedDeviceProperties.memoryBusWidth  << " bits" << endl;
	cout << "Peak memory bandwidth:      " << 2.0*SelectedDeviceProperties.memoryClockRate*(SelectedDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << endl;
	cout << endl;
	cout << "Warp size:                  " << SelectedDeviceProperties.warpSize << endl;
	cout << "Max. warps per multiproc:   " << SelectedDeviceProperties.maxThreadsPerMultiProcessor / SelectedDeviceProperties.warpSize << endl;
	cout << "Max. threads per multiproc: " << SelectedDeviceProperties.maxThreadsPerMultiProcessor << endl;
	cout << "Max. threads per block:     " << SelectedDeviceProperties.maxThreadsPerBlock << endl;
	cout << "Max. block dimensions:      " << SelectedDeviceProperties.maxThreadsDim[0] << " * " << SelectedDeviceProperties.maxThreadsDim[1] << " * " << SelectedDeviceProperties.maxThreadsDim[2] << endl;
	cout << "Max. grid dimensions:       " << SelectedDeviceProperties.maxGridSize[0] << " * " << SelectedDeviceProperties.maxGridSize[1] << " * " << SelectedDeviceProperties.maxGridSize[2] << endl;
	cout << endl;
	cout << "Concurrent memory copy:     " << SelectedDeviceProperties.deviceOverlap << endl;
	cout << "Execution multiple kernels: " << SelectedDeviceProperties.concurrentKernels << endl;
	cout << "ECC support turned on:      " << SelectedDeviceProperties.ECCEnabled << endl << endl;
	
	cout << endl;
}

/*void CheckStorageRequirements(const ConstructorConfiguration& Configuration, int SelectedDevice)
{
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, SelectedDevice);
	
	std::streamsize DefaultPrecision = std::cout.precision();
	
	// Check constant memory usage
	int ConstantMemoryRequired  = 27*sizeof(double);
	int ConstantMemoryAvailable = SelectedDeviceProperties.totalConstMem;
		cout << "Constant memory required RKCK45:     "; cout.width(6); cout << ConstantMemoryRequired+464+372 << " b" << endl;
		cout << "Constant memory required RKCK45_EH0: "; cout.width(6); cout << ConstantMemoryRequired+464+372 << " b" << endl;
		cout << "Constant memory required RK4:        "; cout.width(6); cout << ConstantMemoryRequired+464+372 << " b" << endl;
		cout << "Constant memory required RK4_EH0:    "; cout.width(6); cout << ConstantMemoryRequired+464+144 << " b" << endl;
		cout << "Constant memory available:           "; cout.width(6); cout << ConstantMemoryAvailable << " b" << endl;
	cout << "---------------------------------------------" << endl;
	
	// Shared memory usage for RK4
	int SharedMemoryRequired_RKCK45 = 2*Configuration.SystemDimension*sizeof(double) + Configuration.NumberOfEvents*sizeof(int) + \
									    Configuration.NumberOfEvents*sizeof(double) + \
									    Configuration.NumberOfEvents*sizeof(int) + \
									    Configuration.NumberOfSharedParameters*sizeof(double);
	int SharedMemoryRequired_RKCK45_EH0 = 2*Configuration.SystemDimension*sizeof(double) + \
										    Configuration.NumberOfSharedParameters*sizeof(double);
	int SharedMemoryRequired_RK4 = Configuration.NumberOfEvents*sizeof(int) + \
								   Configuration.NumberOfEvents*sizeof(double) + \
								   Configuration.NumberOfEvents*sizeof(int) + \
								   Configuration.NumberOfSharedParameters*sizeof(double);
	int SharedMemoryRequired_RK4_EH0 = Configuration.NumberOfSharedParameters*sizeof(double);
	
	int SharedMemoryAvailable     = SelectedDeviceProperties.sharedMemPerBlock;
		
		cout << "Shared memory required RKCK45:     "; cout.width(8); cout << SharedMemoryRequired_RKCK45 << " b" << endl;
		cout << "Shared memory required RKCK45_EH0: "; cout.width(8); cout << SharedMemoryRequired_RKCK45_EH0 << " b" << endl;
		cout << "Shared memory required RK4:        "; cout.width(8); cout << SharedMemoryRequired_RK4 << " b" << endl;
		cout << "Shared memory required RK4_EH0:    "; cout.width(8); cout << SharedMemoryRequired_RK4_EH0 << " b" << endl;
		cout << "Shared memory available:           "; cout.width(8); cout << SharedMemoryAvailable << " b" << endl << endl;
		
		cout << "Number of possible blocks / SM RKCK45:     "; cout.width(6); cout << static_cast<int>( (double)SharedMemoryAvailable / SharedMemoryRequired_RKCK45 ) << endl;
		cout << "Number of possible blocks / SM RKCK45_EH0: "; cout.width(6); cout << static_cast<int>( (double)SharedMemoryAvailable / SharedMemoryRequired_RKCK45_EH0 ) << endl;
		if (SharedMemoryRequired_RK4 > 0)
		{
			cout << "Number of possible blocks / SM RK4:        "; cout.width(6); cout << static_cast<int>( (double)SharedMemoryAvailable / SharedMemoryRequired_RK4 ) << endl;
		} else
		{
			cout << "Number of possible blocks / SM RK4:        "; cout.width(6); cout << "inf" << endl;
		}
		if (SharedMemoryRequired_RK4_EH0 > 0)
		{
			cout << "Number of possible blocks / SM RK4_EH0:    "; cout.width(6); cout << static_cast<int>( (double)SharedMemoryAvailable / SharedMemoryRequired_RK4_EH0 ) << endl;
		} else
		{
			cout << "Number of possible blocks / SM RK4_EH0:    "; cout.width(6); cout << "inf" << endl;
		}
		
	cout << "-------------------------------------------------" << endl;
	
	// Global memory usage
	int GlobalMemoryRequired  = (2 + 12*Configuration.SystemDimension + Configuration.NumberOfControlParameters + Configuration.NumberOfAccessories) * Configuration.NumberOfThreads * sizeof(double) + \
	                            Configuration.NumberOfSharedParameters * Configuration.NumberOfThreads * sizeof(double) + \
								10*Configuration.SystemDimension * Configuration.NumberOfThreads * sizeof(double) + \
	                            (2*sizeof(double) + 2*sizeof(int)) * Configuration.NumberOfEvents * Configuration.NumberOfThreads;
	int GlobalMemoryAvailable = SelectedDeviceProperties.totalGlobalMem;
	
	double GlobalMemoryUsageRatio = (double) GlobalMemoryRequired / GlobalMemoryAvailable;
		cout << "Global memory required:  "; cout.width(10); cout << GlobalMemoryRequired  / 1024 << " Kb" << endl;
		cout << "Global memory available: "; cout.width(10); cout << GlobalMemoryAvailable / 1024 << " Kb" << endl;
		cout << "Global memory usage:     "; cout.width(10); cout << setprecision(1) << 100 * GlobalMemoryUsageRatio << " %" << endl;
		cout << setprecision(DefaultPrecision);
	cout << "--------------------------------------" << endl;
	
	cout << endl;
}*/

// --- PROBLEM SOLVER OBJECT ---

ProblemSolver::ProblemSolver(const ConstructorConfiguration& Configuration, int AssociatedDevice)
{
    Device = AssociatedDevice;
	gpuErrCHK( cudaSetDevice(Device) );
	
	gpuErrCHK( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
	gpuErrCHK( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
	
	gpuErrCHK( cudaStreamCreate(&Stream) );
	gpuErrCHK( cudaEventCreate(&Event) );
	
	
	KernelParameters.UnitDimension   = Configuration.UnitDimension;
	KernelParameters.UnitsPerSystem  = Configuration.UnitsPerSystem;
	KernelParameters.NumberOfSystems = Configuration.NumberOfSystems;
	KernelParameters.SystemPerBlock  = Configuration.SystemPerBlock;
	KernelParameters.ThreadsPerBlock = Configuration.ThreadsPerBlock;
	
	KernelParameters.NumberOfControlParameters = Configuration.NumberOfControlParameters;
	KernelParameters.NumberOfSharedParameters  = Configuration.NumberOfSharedParameters;
	KernelParameters.NumberOfEvents            = Configuration.NumberOfEvents;
	KernelParameters.NumberOfAccessories       = Configuration.NumberOfAccessories;
	
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
	
	
	SizeOfTimeDomain        = KernelParameters.NumberOfSystems * 2;
	SizeOfActualState       = KernelParameters.ThreadAllocationRequired * KernelParameters.UnitDimension;
	SizeOfControlParameters = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfControlParameters;
	SizeOfSharedParameters  = KernelParameters.NumberOfSharedParameters;
	SizeOfAccessories       = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfAccessories;
	SizeOfEvents            = KernelParameters.ThreadAllocationRequired * KernelParameters.NumberOfEvents;
	SizeOfCouplingMatrix    = KernelParameters.UnitsPerSystem * KernelParameters.UnitsPerSystem;
	
	h_TimeDomain        = AllocateHostPinnedMemory<double>( SizeOfTimeDomain );
	h_ActualState       = AllocateHostPinnedMemory<double>( SizeOfActualState );
	h_ControlParameters = AllocateHostPinnedMemory<double>( SizeOfControlParameters );
	h_SharedParameters  = AllocateHostPinnedMemory<double>( SizeOfSharedParameters );
	h_Accessories       = AllocateHostPinnedMemory<double>( SizeOfAccessories );
	h_CouplingMatrix    = AllocateHostPinnedMemory<double>( SizeOfCouplingMatrix );
	
	KernelParameters.d_TimeDomain        = AllocateDeviceMemory<double>( SizeOfTimeDomain );
	KernelParameters.d_ActualState       = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_ControlParameters = AllocateDeviceMemory<double>( SizeOfControlParameters );
	KernelParameters.d_SharedParameters  = AllocateDeviceMemory<double>( SizeOfSharedParameters );
	KernelParameters.d_Accessories       = AllocateDeviceMemory<double>( SizeOfAccessories );
	KernelParameters.d_CouplingMatrix    = AllocateDeviceMemory<double>( SizeOfCouplingMatrix );
	
	KernelParameters.d_State    = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_Stages   = AllocateDeviceMemory<double>( SizeOfActualState * 6 );
	
	KernelParameters.d_NextState = AllocateDeviceMemory<double>( SizeOfActualState );
	
	KernelParameters.d_Error           = AllocateDeviceMemory<double>( SizeOfActualState );
	KernelParameters.d_ActualTolerance = AllocateDeviceMemory<double>( SizeOfActualState );
	
	KernelParameters.d_ActualEventValue        = AllocateDeviceMemory<double>( SizeOfEvents );
	KernelParameters.d_NextEventValue          = AllocateDeviceMemory<double>( SizeOfEvents );
	KernelParameters.d_EventCounter            = AllocateDeviceMemory<int>(    SizeOfEvents );
	KernelParameters.d_EventEquilibriumCounter = AllocateDeviceMemory<int>(    SizeOfEvents );
	
	
	//DynamicSharedMemoryRKCK45     = 2*KernelParameters.UnitDimension*sizeof(double) + KernelParameters.NumberOfEvents*( sizeof(int) + sizeof(double) + sizeof(int) ) + KernelParameters.NumberOfSharedParameters*sizeof(double);
	//DynamicSharedMemoryRKCK45_EH0 = 2*KernelParameters.UnitDimension*sizeof(double) + KernelParameters.NumberOfSharedParameters*sizeof(double);
	//DynamicSharedMemoryRK4        = KernelParameters.NumberOfEvents*( sizeof(int) + sizeof(double) + sizeof(int) ) + KernelParameters.NumberOfSharedParameters*sizeof(double) + KernelParameters.ThreadsPerBlockRequired*sizeof(double);
	DynamicSharedMemory_RK4_EH0_SSSBL = KernelParameters.NumberOfSharedParameters * sizeof(double) + \
	                                    2*KernelParameters.ThreadsPerBlockRequired * sizeof(double) + \
										(2+1+1+1)*KernelParameters.SystemPerBlock * sizeof(double) + \
										KernelParameters.SystemPerBlock * sizeof(int);
	
	
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
	
	// Default settings
	KernelParameters.InitialTimeStep = 1e-2;
	KernelParameters.ActiveSystems   = KernelParameters.NumberOfSystems;
	
	Solver = RKCK45;
	
	cout << "Object for Parameters scan is successfully created! Required memory allocations have been done" << endl << endl;
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
	
	cout << "Object for Parameters scan is deleted! Every memory have been deallocated!" << endl << endl;
}

// Unit scope
void ProblemSolver::SetHost(int SystemNumber, int UnitNumber, VariableSelection ActualVariable, int SerialNumber, double Value)
{
	if ( SystemNumber >= KernelParameters.NumberOfSystems )
	{
        cerr << "ERROR in solver member function SetHost:" << endl << "    The index of the coupled system cannot be larger than " << KernelParameters.NumberOfSystems-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( UnitNumber >= KernelParameters.UnitsPerSystem )
	{
        cerr << "ERROR in solver member function SetHost:" << endl << "    The index of the unit in a coupled system cannot be larger than " << KernelParameters.UnitsPerSystem-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int BlockSerialNumber         = SystemNumber / KernelParameters.SystemPerBlock;
	int SystemSerialNumberInBlock = SystemNumber % KernelParameters.SystemPerBlock;
	
	int tid = BlockSerialNumber*(KernelParameters.ThreadsPerBlockRequired+KernelParameters.ThreadPadding) + SystemSerialNumberInBlock*KernelParameters.UnitsPerSystem + UnitNumber;
	int gid = tid + SerialNumber*KernelParameters.ThreadAllocationRequired;
	
	
	switch (ActualVariable)
	{
		case ActualState:
			if ( SerialNumber >= KernelParameters.UnitDimension )
			{
				cerr << "ERROR in solver member function SetHost:" << endl << "    The serial number of the ActualState cannot be larger than " << KernelParameters.UnitDimension-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_ActualState[gid] = Value;
			break;
			
		case ControlParameters:
			if ( SerialNumber >= KernelParameters.NumberOfControlParameters )
			{
				cerr << "ERROR in solver member function SetHost:" << endl << "    The serial number of the ControlParameters cannot be larger than " << KernelParameters.NumberOfControlParameters-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_ControlParameters[gid] = Value;
			break;
			
		case Accessories:
			if ( SerialNumber >= KernelParameters.NumberOfAccessories )
			{
				cerr << "ERROR in solver member function SetHost:" << endl << "    The serial number of the Accessories cannot be larger than " << KernelParameters.NumberOfAccessories-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_Accessories[gid] = Value;
			break;
			
		default :
			cerr << "ERROR in solver member function SetHost:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// System scope
void ProblemSolver::SetHost(int SystemNumber, VariableSelection ActualVariable, int SerialNumber, double Value)
{
	if ( SystemNumber >= KernelParameters.NumberOfSystems )
	{
        cerr << "ERROR in solver member function SetHost:" << endl << "    The index of the coupled system cannot be larger than " << KernelParameters.NumberOfSystems-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int sid = SystemNumber + SerialNumber*KernelParameters.NumberOfSystems;
	
	switch (ActualVariable)
	{
		case TimeDomain:
			if ( SerialNumber >= 2 )
			{
				cerr << "ERROR in solver member function SetHost:" << endl << "    The serial number of the TimeDomain cannot be larger than " << 2-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			h_TimeDomain[sid] = Value;
			break;
		
		default :
			cerr << "ERROR in solver member function SetHost:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Global scope
void ProblemSolver::SetHost(VariableSelection ActualVariable, int SerialNumber, double Value)
{
	if ( SerialNumber >= KernelParameters.NumberOfSharedParameters )
	{
        cerr << "ERROR in solver member function SetHost:" << endl << "    The serial number of the SharedParameters cannot be larger than " << KernelParameters.NumberOfSharedParameters-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( ActualVariable != SharedParameters )
	{
		cerr << "ERROR in solver member function SetHost:" << endl << "    Invalid option for variable selection!\n";
		exit(EXIT_FAILURE);
	}
	
	h_SharedParameters[SerialNumber] = Value;
}

// Coupling matrix
void ProblemSolver::SetHost(VariableSelection ActualVariable, int Row, int Col, double Value)
{
	if ( ( Row >= KernelParameters.UnitsPerSystem ) || ( Col >= KernelParameters.UnitsPerSystem ) )
	{
        cerr << "ERROR in solver member function SetHost:" << endl << "    The Row or Column of the CouplingMatrix cannot be larger than " << KernelParameters.UnitsPerSystem-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( ActualVariable != CouplingMatrix )
	{
		cerr << "ERROR in solver member function SetHost:" << endl << "    Invalid option for variable selection!\n";
		exit(EXIT_FAILURE);
	}
	
	int idx = Row + Col*KernelParameters.UnitsPerSystem;
	h_CouplingMatrix[idx] = Value;
}

void ProblemSolver::SynchroniseFromHostToDevice(VariableSelection ActualVariable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (ActualVariable)
	{
		case TimeDomain:        gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_TimeDomain,        h_TimeDomain,               SizeOfTimeDomain * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		case ActualState:       gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ActualState,       h_ActualState,             SizeOfActualState * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		case ControlParameters: gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ControlParameters, h_ControlParameters, SizeOfControlParameters * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		case SharedParameters:  gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_SharedParameters,  h_SharedParameters,   SizeOfSharedParameters * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		case Accessories:       gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_Accessories,       h_Accessories,             SizeOfAccessories * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		case CouplingMatrix:    gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_CouplingMatrix,    h_CouplingMatrix,       SizeOfCouplingMatrix * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		case All: gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_TimeDomain,        h_TimeDomain,               SizeOfTimeDomain * sizeof(double), cudaMemcpyHostToDevice, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ActualState,       h_ActualState,             SizeOfActualState * sizeof(double), cudaMemcpyHostToDevice, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_ControlParameters, h_ControlParameters, SizeOfControlParameters * sizeof(double), cudaMemcpyHostToDevice, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_SharedParameters,  h_SharedParameters,   SizeOfSharedParameters * sizeof(double), cudaMemcpyHostToDevice, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_Accessories,       h_Accessories,             SizeOfAccessories * sizeof(double), cudaMemcpyHostToDevice, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(KernelParameters.d_CouplingMatrix,    h_CouplingMatrix,       SizeOfCouplingMatrix * sizeof(double), cudaMemcpyHostToDevice, Stream) ); break;
		default :
			cerr << "ERROR in solver member function SynchroniseFromHostToDevice:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

void ProblemSolver::SynchroniseFromDeviceToHost(VariableSelection ActualVariable)
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	switch (ActualVariable)
	{
		case TimeDomain:        gpuErrCHK( cudaMemcpyAsync(h_TimeDomain,        KernelParameters.d_TimeDomain,               SizeOfTimeDomain * sizeof(double), cudaMemcpyDeviceToHost, Stream) ); break;
		case ActualState:       gpuErrCHK( cudaMemcpyAsync(h_ActualState,       KernelParameters.d_ActualState,             SizeOfActualState * sizeof(double), cudaMemcpyDeviceToHost, Stream) ); break;
		case ControlParameters: gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, KernelParameters.d_ControlParameters, SizeOfControlParameters * sizeof(double), cudaMemcpyDeviceToHost, Stream) ); break;
		case SharedParameters:  gpuErrCHK( cudaMemcpyAsync(h_SharedParameters,  KernelParameters.d_SharedParameters,   SizeOfSharedParameters * sizeof(double), cudaMemcpyDeviceToHost, Stream) ); break;
		case Accessories:       gpuErrCHK( cudaMemcpyAsync(h_Accessories,       KernelParameters.d_Accessories,             SizeOfAccessories * sizeof(double), cudaMemcpyDeviceToHost, Stream) ); break;
		case All: gpuErrCHK( cudaMemcpyAsync(h_TimeDomain,        KernelParameters.d_TimeDomain,               SizeOfTimeDomain * sizeof(double), cudaMemcpyDeviceToHost, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(h_ActualState,       KernelParameters.d_ActualState,             SizeOfActualState * sizeof(double), cudaMemcpyDeviceToHost, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(h_ControlParameters, KernelParameters.d_ControlParameters, SizeOfControlParameters * sizeof(double), cudaMemcpyDeviceToHost, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(h_SharedParameters,  KernelParameters.d_SharedParameters,   SizeOfSharedParameters * sizeof(double), cudaMemcpyDeviceToHost, Stream) );
				  gpuErrCHK( cudaMemcpyAsync(h_Accessories,       KernelParameters.d_Accessories,             SizeOfAccessories * sizeof(double), cudaMemcpyDeviceToHost, Stream) ); break;
		default :
			cerr << "ERROR in solver member function SynchroniseFromDeviceToHost:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
}

// Unit scope
double ProblemSolver::GetHost(int SystemNumber, int UnitNumber, VariableSelection ActualVariable, int SerialNumber)
{
	if ( SystemNumber >= KernelParameters.NumberOfSystems )
	{
        cerr << "ERROR in solver member function GetHost:" << endl << "    The index of the coupled system cannot be larger than " << KernelParameters.NumberOfSystems-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( UnitNumber >= KernelParameters.UnitsPerSystem )
	{
        cerr << "ERROR in solver member function GetHost:" << endl << "    The index of the unit in a coupled system cannot be larger than " << KernelParameters.UnitsPerSystem-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int BlockSerialNumber         = SystemNumber / KernelParameters.SystemPerBlock;
	int SystemSerialNumberInBlock = SystemNumber % KernelParameters.SystemPerBlock;
	
	int tid = BlockSerialNumber*(KernelParameters.ThreadsPerBlockRequired+KernelParameters.ThreadPadding) + SystemSerialNumberInBlock*KernelParameters.UnitsPerSystem + UnitNumber;
	int gid = tid + SerialNumber*KernelParameters.ThreadAllocationRequired;
	
	
	double Value;
	switch (ActualVariable)
	{
		case ActualState:
			if ( SerialNumber >= KernelParameters.UnitDimension )
			{
				cerr << "ERROR in solver member function GetHost:" << endl << "    The serial number of the ActualState cannot be larger than " << KernelParameters.UnitDimension-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_ActualState[gid];
			break;
		case ControlParameters:
			if ( SerialNumber >= KernelParameters.NumberOfControlParameters )
			{
				cerr << "ERROR in solver member function GetHost:" << endl << "    The serial number of the ControlParameters cannot be larger than " << KernelParameters.NumberOfControlParameters-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_ControlParameters[gid];
			break;
		case Accessories:
			if ( SerialNumber >= KernelParameters.NumberOfAccessories )
			{
				cerr << "ERROR in solver member function GetHost:" << endl << "    The serial number of the Accessories cannot be larger than " << KernelParameters.NumberOfAccessories-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_Accessories[gid];
			break;
		
		default :
			cerr << "ERROR in solver member function GetHost:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// System scope
double ProblemSolver::GetHost(int SystemNumber, VariableSelection ActualVariable, int SerialNumber)
{
	if ( SystemNumber >= KernelParameters.NumberOfSystems )
	{
        cerr << "ERROR in solver member function GetHost:" << endl << "    The index of the coupled system cannot be larger than " << KernelParameters.NumberOfSystems-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	int sid = SystemNumber + SerialNumber*KernelParameters.NumberOfSystems;
	
	double Value;
	switch (ActualVariable)
	{
		case TimeDomain:
			if ( SerialNumber >= 2 )
			{
				cerr << "ERROR in solver member function GetHost:" << endl << "    The serial number of the TimeDomain cannot be larger than " << 2-1 << "! (The indexing starts from zero)\n";
				exit(EXIT_FAILURE);
			}
			Value = h_TimeDomain[sid];
			break;
		
		default :
			cerr << "ERROR in solver member function GetHost:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
	
	return Value;
}

// Global scope
double ProblemSolver::GetHost(VariableSelection ActualVariable, int SerialNumber)
{
	if ( SerialNumber >= KernelParameters.NumberOfSharedParameters )
	{
        cerr << "ERROR in solver member function GetHost:" << endl << "    The serial number of the SharedParameters cannot be larger than " << KernelParameters.NumberOfSharedParameters-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( ActualVariable != SharedParameters )
	{
		cerr << "ERROR in solver member function GetHost:" << endl << "    Invalid option for variable selection!\n";
		exit(EXIT_FAILURE);
	}
	
	double Value = h_SharedParameters[SerialNumber];
	
	return Value;
}

// Coupling matrix
double ProblemSolver::GetHost(VariableSelection ActualVariable, int Row, int Col)
{
	if ( ( Row >= KernelParameters.UnitsPerSystem ) || ( Col >= KernelParameters.UnitsPerSystem ) )
	{
        cerr << "ERROR in solver member function GetHost:" << endl << "    The Row or Column of the CouplingMatrix cannot be larger than " << KernelParameters.UnitsPerSystem-1 << "! (The indexing starts from zero)\n";
        exit(EXIT_FAILURE);
    }
	
	if ( ActualVariable != CouplingMatrix )
	{
		cerr << "ERROR in solver member function GetHost:" << endl << "    Invalid option for variable selection!\n";
		exit(EXIT_FAILURE);
	}
	
	int idx = Row + Col*KernelParameters.UnitsPerSystem;
	double Value = h_CouplingMatrix[idx];
	
	return Value;
}

void ProblemSolver::Print(VariableSelection ActualVariable)
{
	ofstream DataFile;
	int NumberOfRows;
	int NumberOfColumns;
	double* PointerToActualData;
	
	switch (ActualVariable)
	{
		case TimeDomain:		DataFile.open ( "TimeDomainInSolverObject.txt" );
								NumberOfRows = KernelParameters.ThreadAllocationRequired;
								NumberOfColumns = 2;
								PointerToActualData = h_TimeDomain; break;
		case ActualState:		DataFile.open ( "ActualStateInSolverObject.txt" );
								NumberOfRows = KernelParameters.ThreadAllocationRequired;
								NumberOfColumns = KernelParameters.UnitDimension;
								PointerToActualData = h_ActualState; break;
		case ControlParameters: DataFile.open ( "ControlParametersInSolverObject.txt" );
								NumberOfRows = KernelParameters.ThreadAllocationRequired;
								NumberOfColumns = KernelParameters.NumberOfControlParameters;
								PointerToActualData = h_ControlParameters; break;
		case SharedParameters:  DataFile.open ( "SharedParametersInSolverObject.txt" );
								NumberOfRows    = KernelParameters.NumberOfSharedParameters;		
								NumberOfColumns = 1;
								PointerToActualData = h_SharedParameters; break;				  
		case Accessories:		DataFile.open ( "AccessoriesInSolverObject.txt" );
								NumberOfRows = KernelParameters.ThreadAllocationRequired;
								NumberOfColumns = KernelParameters.NumberOfAccessories;
								PointerToActualData = h_Accessories; break;
		case CouplingMatrix:	DataFile.open ( "CouplingMatrixInSolverObject.txt" );
								NumberOfRows = KernelParameters.UnitsPerSystem;
								NumberOfColumns = KernelParameters.UnitsPerSystem;
								PointerToActualData = h_CouplingMatrix; break;
		
		default :
			cerr << "ERROR in solver member function Print:" << endl << "    Invalid option for variable selection!\n";
			exit(EXIT_FAILURE);
	}
		
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	int idx;
	int GlobalLimit;
	if ( (ActualVariable==SharedParameters) || (ActualVariable==CouplingMatrix) )
	{
		for (int i=0; i<NumberOfRows; i++)
		{
			for (int j=0; j<NumberOfColumns; j++)
			{
				idx = i + j*NumberOfRows;
				DataFile.width(Width); DataFile << PointerToActualData[idx] << ',';
			}
			DataFile << '\n';
		}
	} else
	{
		for (int i=0; i<NumberOfRows; i++)
		{
			for (int j=0; j<NumberOfColumns; j++)
			{
				idx = i + j*NumberOfRows;
				
				GlobalLimit = (KernelParameters.NumberOfSystems / KernelParameters.SystemPerBlock) * (KernelParameters.ThreadsPerBlockRequired+KernelParameters.ThreadPadding) + \
				              (KernelParameters.NumberOfSystems % KernelParameters.SystemPerBlock) * KernelParameters.UnitsPerSystem - 1;
				
				if ( ( i%(KernelParameters.ThreadsPerBlockRequired+KernelParameters.ThreadPadding) > (KernelParameters.ThreadsPerBlockRequired-1) ) || ( i>GlobalLimit ) )
				{
					DataFile.width(Width); DataFile << "PADDING" << ',';
				} else
				{
					DataFile.width(Width); DataFile << PointerToActualData[idx] << ',';
				}
			}
			DataFile << '\n';
		}
	}
}

void ProblemSolver::SolverOption(ListOfSolverAlgorithms Algorithm, double InitialTimeStep, int ActiveSystems)
{
	KernelParameters.InitialTimeStep = InitialTimeStep;
	KernelParameters.ActiveSystems   = ActiveSystems;
	Solver = Algorithm;
}

void ProblemSolver::Solve()
{
	gpuErrCHK( cudaSetDevice(Device) );
	
	int GridSize  = KernelParameters.GridSize;
	int BlockSize = KernelParameters.ThreadsPerBlock;
	
	
	//if ( Configuration.Solver==RKCK45 )
	//	PerThread_RKCK45<<<GridSize, Configuration.BlockSize, DynamicSharedMemoryRKCK45, Stream>>> (KernelParameters);
	
	//if ( Configuration.Solver==RKCK45_EH0 )
	//	PerThread_RKCK45_EH0<<<GridSize, Configuration.BlockSize, DynamicSharedMemoryRKCK45_EH0, Stream>>> (KernelParameters);
	
	//if ( Solver==RK4 )
	//	PerBlockCoupling_RK4<<<GridSize, BlockSize, DynamicSharedMemoryRK4, Stream>>> (KernelParameters);
	
	//if ( Solver==RK4_EH0 )
	//	PerBlockCoupling_RK4_EH0<<<GridSize, BlockSize, DynamicSharedMemoryRK4_EH0, Stream>>> (KernelParameters);

	if ( Solver==RK4_EH0_SSSBL )
	{
		PerBlockCoupling_RK4_EH0_SSSBL<<<GridSize, BlockSize, DynamicSharedMemory_RK4_EH0_SSSBL, Stream>>> (KernelParameters);
	}
		
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
        cerr << "Failed to allocate Memory on the DEVICE!\n";
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
        cerr << "Failed to allocate Pinned Memory on the HOST!\n";
        exit(EXIT_FAILURE);
    }
    return MemoryAddressInHost;
}

template <class DataType>
DataType* AllocateHostMemory(int N)
{
    DataType* HostMemory = new (nothrow) DataType [N];
    if (HostMemory == NULL)
    {
        cerr << "Failed to allocate Memory on the HOST!\n";
        exit(EXIT_FAILURE);
    }
    return HostMemory;
}
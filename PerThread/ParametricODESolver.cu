#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>

#include "ParametricODESolver.cuh"

#include "ParametricODESystem.cuh"
#include "ParametricODERungeKutta.cuh"

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

void PrintPropertiesOfTheSelectedDevice(int SelectedDevice)
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
	cout << "Max. threads per block:     " << SelectedDeviceProperties.maxThreadsPerBlock << endl;
	cout << "Max. block dimensions:      " << SelectedDeviceProperties.maxThreadsDim[0] << " * " << SelectedDeviceProperties.maxThreadsDim[1] << " * " << SelectedDeviceProperties.maxThreadsDim[2] << endl;
	cout << "Max. grid dimensions:       " << SelectedDeviceProperties.maxGridSize[0] << " * " << SelectedDeviceProperties.maxGridSize[1] << " * " << SelectedDeviceProperties.maxGridSize[2] << endl;
	cout << endl;
	cout << "Concurrent memory copy:     " << SelectedDeviceProperties.deviceOverlap << endl;
	cout << "Execution multiple kernels: " << SelectedDeviceProperties.concurrentKernels << endl;
	cout << "ECC support turned on:      " << SelectedDeviceProperties.ECCEnabled << endl << endl;
	
	cout << endl;
}

void CheckStorageRequirements(const ConstructorConfiguration& Configuration, int SelectedDevice)
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
		cout << "Number of possible blocks / SM RK4:        "; cout.width(6); cout << static_cast<int>( (double)SharedMemoryAvailable / SharedMemoryRequired_RK4 ) << endl;
		cout << "Number of possible blocks / SM RK4_EH0:    "; cout.width(6); cout << static_cast<int>( (double)SharedMemoryAvailable / SharedMemoryRequired_RK4_EH0 ) << endl;
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
}

// --- PROBLEM POOL OBJECT ---

ProblemPool::ProblemPool(const ConstructorConfiguration& Configuration)
{
    PoolSize = Configuration.PoolSize;
	
	SystemDimension           = Configuration.SystemDimension;
	NumberOfControlParameters = Configuration.NumberOfControlParameters;
	NumberOfSharedParameters  = Configuration.NumberOfSharedParameters;
	NumberOfAccessories       = Configuration.NumberOfAccessories;
	
	p_TimeDomain        = AllocateHostMemory<double>( PoolSize * 2 );
	p_ActualState       = AllocateHostMemory<double>( PoolSize * SystemDimension );
	p_ControlParameters = AllocateHostMemory<double>( PoolSize * NumberOfControlParameters );
	p_SharedParameters  = AllocateHostMemory<double>( NumberOfSharedParameters );
	p_Accessories       = AllocateHostMemory<double>( PoolSize * NumberOfAccessories );
	
	cout << "Object for problem pool is successfully created! Required memory allocations have been done" << endl << endl;
}

ProblemPool::~ProblemPool()
{
    delete[] p_TimeDomain;
	delete[] p_ActualState;
	delete[] p_ControlParameters;
	delete[] p_SharedParameters;
	delete[] p_Accessories;
	
	cout << "Object for problem pool is deleted! Every memory have been deallocated!" << endl << endl;
}

void ProblemPool::Set(int ProblemNumber, VariableSelection ActualVariable, int SerialNumber, double Value)
{
	int idx = ProblemNumber + SerialNumber*PoolSize;
	
	switch (ActualVariable)
	{
		case TimeDomain:        p_TimeDomain[idx]        = Value; break;
		case ActualState:       p_ActualState[idx]       = Value; break;
		case ControlParameters: p_ControlParameters[idx] = Value; break;
		case Accessories:       p_Accessories[idx]       = Value; break;
	}
}

void ProblemPool::SetShared(int SerialNumber, double Value)
{
	p_SharedParameters[SerialNumber] = Value;
}

double ProblemPool::Get(int ProblemNumber, VariableSelection ActualVariable, int SerialNumber)
{
	int idx = ProblemNumber + SerialNumber*PoolSize;
	
	double Value;
	switch (ActualVariable)
	{
		case TimeDomain:        Value = p_TimeDomain[idx];        break;
		case ActualState:       Value = p_ActualState[idx];       break;
		case ControlParameters: Value = p_ControlParameters[idx]; break;
		case Accessories:       Value = p_Accessories[idx];       break;
	}
	
	return Value;
}

double ProblemPool::GetShared(int SerialNumber)
{
	double Value = p_SharedParameters[SerialNumber];
	
	return Value;
}

void ProblemPool::Print(VariableSelection ActualVariable)
{
	ofstream DataFile;
	int NumberOfRows;
	int NumberOfColumns;
	double* PointerToActualData;
	
	switch (ActualVariable)
	{
		case TimeDomain:	    DataFile.open ( "TimeDomainInPool.txt" );
								NumberOfRows    = PoolSize;
								NumberOfColumns = 2;
								PointerToActualData = p_TimeDomain; break;
		case ActualState:	    DataFile.open ( "ActualStateInPool.txt" );
								NumberOfRows    = PoolSize;
								NumberOfColumns = SystemDimension;
								PointerToActualData = p_ActualState; break;
		case ControlParameters: DataFile.open ( "ControlParametersInPool.txt" ); 
								NumberOfRows    = PoolSize;		
								NumberOfColumns = NumberOfControlParameters;
								PointerToActualData = p_ControlParameters; break;
		case SharedParameters:  DataFile.open ( "SharedParametersInPool.txt" ); 
								NumberOfRows    = NumberOfSharedParameters;		
								NumberOfColumns = 1;
								PointerToActualData = p_SharedParameters; break;
		case Accessories:		DataFile.open ( "AccessoriesInPool.txt" );
								NumberOfRows    = PoolSize;
								NumberOfColumns = NumberOfAccessories;
								PointerToActualData = p_Accessories; break;
	}
		
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	int idx;
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			idx = i + j*NumberOfRows;
			DataFile.width(Width); DataFile << PointerToActualData[idx] << ',';
		}
		DataFile << '\n';
	}
}

// --- PROBLEM SOLVER OBJECT ---

ProblemSolver::ProblemSolver(const ConstructorConfiguration& Configuration)
{
    KernelParameters.NumberOfThreads     = Configuration.NumberOfThreads;
	
	KernelParameters.SystemDimension           = Configuration.SystemDimension;
	KernelParameters.NumberOfControlParameters = Configuration.NumberOfControlParameters;
	KernelParameters.NumberOfSharedParameters  = Configuration.NumberOfSharedParameters;
	KernelParameters.NumberOfEvents            = Configuration.NumberOfEvents;
	KernelParameters.NumberOfAccessories       = Configuration.NumberOfAccessories;
	
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
	
	
	h_TimeDomain        = AllocateHostMemory<double>( KernelParameters.NumberOfThreads * 2 );
	h_ActualState       = AllocateHostMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension );
	h_ControlParameters = AllocateHostMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfControlParameters );
	h_SharedParameters  = AllocateHostMemory<double>( KernelParameters.NumberOfSharedParameters );
	h_Accessories       = AllocateHostMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfAccessories );
	
	
	KernelParameters.d_TimeDomain        = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * 2 );
	KernelParameters.d_ActualState       = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension );
	KernelParameters.d_ControlParameters = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfControlParameters );
	KernelParameters.d_SharedParameters  = AllocateDeviceMemory<double>( KernelParameters.NumberOfSharedParameters );
	KernelParameters.d_Accessories       = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfAccessories );
	
	KernelParameters.d_State    = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension );
	KernelParameters.d_Stages   = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension * 6 );
	
	KernelParameters.d_NextState = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension );
	
	KernelParameters.d_Error           = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension );
	KernelParameters.d_ActualTolerance = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.SystemDimension );
	
	KernelParameters.d_ActualEventValue        = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfEvents );
	KernelParameters.d_NextEventValue          = AllocateDeviceMemory<double>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfEvents );
	KernelParameters.d_EventCounter            = AllocateDeviceMemory<int>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfEvents );
	KernelParameters.d_EventEquilibriumCounter = AllocateDeviceMemory<int>( KernelParameters.NumberOfThreads * KernelParameters.NumberOfEvents );
	
	KernelParameters.InitialTimeStep = 0;
	
	cout << "Object for Parameters scan is successfully created! Required memory allocations have been done" << endl << endl;
}

ProblemSolver::~ProblemSolver()
{
    delete[] h_TimeDomain;
	delete[] h_ActualState;
	delete[] h_ControlParameters;
	delete[] h_SharedParameters;
	delete[] h_Accessories;
	
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

void ProblemSolver::LinearCopyFromPoolHostAndDevice(const ProblemPool& Pool, int CopyStartIndexInPool, int CopyStartIndexInSolverObject, int NumberOfElementsCopied, VariableSelection CopyMode)
{
	double* h_CopyStart;
	double* d_CopyStart;
	double* p_CopyStart;
	
	int CopySize = NumberOfElementsCopied * sizeof(double);
	
	if ( (CopyMode==All) || (CopyMode==TimeDomain) )
	{
		for (int i=0; i<2; i++)
		{
			h_CopyStart = h_TimeDomain + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			d_CopyStart = KernelParameters.d_TimeDomain + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			
			p_CopyStart = Pool.p_TimeDomain + CopyStartIndexInPool + i*Pool.PoolSize;
			
			memcpy(h_CopyStart, p_CopyStart, CopySize);
			gpuErrCHK( cudaMemcpy(d_CopyStart, h_CopyStart, CopySize, cudaMemcpyHostToDevice) );
		}
	}
	
	if ( (CopyMode==All) || (CopyMode==ActualState) )
	{
		for (int i=0; i<KernelParameters.SystemDimension; i++)
		{
			h_CopyStart = h_ActualState + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			d_CopyStart = KernelParameters.d_ActualState + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			
			p_CopyStart = Pool.p_ActualState + CopyStartIndexInPool + i*Pool.PoolSize;
			
			memcpy(h_CopyStart, p_CopyStart, CopySize);
			gpuErrCHK( cudaMemcpy(d_CopyStart, h_CopyStart, CopySize, cudaMemcpyHostToDevice) );
		}
	}
	
	if ( (CopyMode==All) || (CopyMode==ControlParameters) )
	{
		for (int i=0; i<KernelParameters.NumberOfControlParameters; i++)
		{
			h_CopyStart = h_ControlParameters + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			d_CopyStart = KernelParameters.d_ControlParameters + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			
			p_CopyStart = Pool.p_ControlParameters + CopyStartIndexInPool + i*Pool.PoolSize;
			
			memcpy(h_CopyStart, p_CopyStart, CopySize);
			gpuErrCHK( cudaMemcpy(d_CopyStart, h_CopyStart, CopySize, cudaMemcpyHostToDevice) );
		}
	}
	
	if ( (CopyMode==All) || (CopyMode==Accessories) )
	{
		for (int i=0; i<KernelParameters.NumberOfAccessories; i++)
		{
			h_CopyStart = h_Accessories + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			d_CopyStart = KernelParameters.d_Accessories + CopyStartIndexInSolverObject + i*KernelParameters.NumberOfThreads;
			
			p_CopyStart = Pool.p_Accessories + CopyStartIndexInPool + i*Pool.PoolSize;
			
			memcpy(h_CopyStart, p_CopyStart, CopySize);
			gpuErrCHK( cudaMemcpy(d_CopyStart, h_CopyStart, CopySize, cudaMemcpyHostToDevice) );
		}
	}
}

void ProblemSolver::SharedCopyFromPoolHostAndDevice(const ProblemPool& Pool)
{
	double* h_CopyStart = h_SharedParameters;
	double* d_CopyStart = KernelParameters.d_SharedParameters;
	double* p_CopyStart = Pool.p_SharedParameters;
	
	int CopySize = KernelParameters.NumberOfSharedParameters * sizeof(double);
	
	memcpy(h_CopyStart, p_CopyStart, CopySize);
	gpuErrCHK( cudaMemcpy(d_CopyStart, h_CopyStart, CopySize, cudaMemcpyHostToDevice) );
}

void ProblemSolver::SingleSetHost(int ProblemNumber, VariableSelection ActualVariable, int SerialNumber, double Value)
{
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	switch (ActualVariable)
	{
		case TimeDomain:        h_TimeDomain[idx]        = Value; break;
		case ActualState:       h_ActualState[idx]       = Value; break;
		case ControlParameters: h_ControlParameters[idx] = Value; break;
		case Accessories:       h_Accessories[idx]       = Value; break;
	}
}

void ProblemSolver::SingleSetHostAndDevice(int ProblemNumber, VariableSelection ActualVariable, int SerialNumber, double Value)
{
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	switch (ActualVariable)
	{
		case TimeDomain: h_TimeDomain[idx] = Value;
						 gpuErrCHK( cudaMemcpy(KernelParameters.d_TimeDomain +idx, h_TimeDomain +idx, sizeof(double), cudaMemcpyHostToDevice) ); break;
		
		case ActualState: h_ActualState[idx] = Value;
						  gpuErrCHK( cudaMemcpy(KernelParameters.d_ActualState+idx, h_ActualState+idx, sizeof(double), cudaMemcpyHostToDevice) ); break;
						  
		case ControlParameters: h_ControlParameters[idx] = Value;
								gpuErrCHK( cudaMemcpy(KernelParameters.d_ControlParameters+idx, h_ControlParameters+idx, sizeof(double), cudaMemcpyHostToDevice) ); break;
		
		case Accessories: h_Accessories[idx] = Value;
						  gpuErrCHK( cudaMemcpy(KernelParameters.d_Accessories+idx, h_Accessories+idx, sizeof(double), cudaMemcpyHostToDevice) ); break;
	}
}

void ProblemSolver::SetSharedHost(int SerialNumber, double Value)
{
	h_SharedParameters[SerialNumber] = Value;
}

void ProblemSolver::SetSharedHostAndDevice(int SerialNumber, double Value)
{
	h_SharedParameters[SerialNumber] = Value;
	
	gpuErrCHK( cudaMemcpy(KernelParameters.d_SharedParameters+SerialNumber, h_SharedParameters+SerialNumber, sizeof(double), cudaMemcpyHostToDevice) );
}

void ProblemSolver::SynchroniseFromHostToDevice(VariableSelection ActualVariable)
{
	switch (ActualVariable)
	{
		case TimeDomain:        gpuErrCHK( cudaMemcpy(KernelParameters.d_TimeDomain,        h_TimeDomain,                                                 2*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) ); break;
		case ActualState:       gpuErrCHK( cudaMemcpy(KernelParameters.d_ActualState,       h_ActualState,                 KernelParameters.SystemDimension*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) ); break;
		case ControlParameters: gpuErrCHK( cudaMemcpy(KernelParameters.d_ControlParameters, h_ControlParameters, KernelParameters.NumberOfControlParameters*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) ); break;
		case Accessories:       gpuErrCHK( cudaMemcpy(KernelParameters.d_Accessories,       h_Accessories,             KernelParameters.NumberOfAccessories*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) ); break;
		
		case All: gpuErrCHK( cudaMemcpy(KernelParameters.d_TimeDomain,        h_TimeDomain,                                                 2*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) );
				  gpuErrCHK( cudaMemcpy(KernelParameters.d_ActualState,       h_ActualState,                 KernelParameters.SystemDimension*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) );
				  gpuErrCHK( cudaMemcpy(KernelParameters.d_ControlParameters, h_ControlParameters, KernelParameters.NumberOfControlParameters*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) );
				  gpuErrCHK( cudaMemcpy(KernelParameters.d_Accessories,       h_Accessories,             KernelParameters.NumberOfAccessories*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyHostToDevice) ); break;
	}
}

void ProblemSolver::SynchroniseFromDeviceToHost(VariableSelection ActualVariable)
{
	switch (ActualVariable)
	{
		case TimeDomain:        gpuErrCHK( cudaMemcpy(h_TimeDomain,        KernelParameters.d_TimeDomain,                                                 2*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) ); break;
		case ActualState:       gpuErrCHK( cudaMemcpy(h_ActualState,       KernelParameters.d_ActualState,                 KernelParameters.SystemDimension*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) ); break;
		case ControlParameters: gpuErrCHK( cudaMemcpy(h_ControlParameters, KernelParameters.d_ControlParameters, KernelParameters.NumberOfControlParameters*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) ); break;
		case Accessories:       gpuErrCHK( cudaMemcpy(h_Accessories,       KernelParameters.d_Accessories,             KernelParameters.NumberOfAccessories*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) ); break;
		
		case All: gpuErrCHK( cudaMemcpy(h_TimeDomain,        KernelParameters.d_TimeDomain,                                                 2*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) );
				  gpuErrCHK( cudaMemcpy(h_ActualState,       KernelParameters.d_ActualState,                 KernelParameters.SystemDimension*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) );
				  gpuErrCHK( cudaMemcpy(h_ControlParameters, KernelParameters.d_ControlParameters, KernelParameters.NumberOfControlParameters*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) );
				  gpuErrCHK( cudaMemcpy(h_Accessories,       KernelParameters.d_Accessories,             KernelParameters.NumberOfAccessories*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) ); break;
	}
}

void ProblemSolver::SynchroniseSharedFromHostToDevice()
{
	gpuErrCHK( cudaMemcpy(KernelParameters.d_SharedParameters, h_SharedParameters, KernelParameters.NumberOfSharedParameters*sizeof(double), cudaMemcpyHostToDevice) );
}

void ProblemSolver::SynchroniseSharedFromDeviceToHost()
{
	gpuErrCHK( cudaMemcpy(h_SharedParameters, KernelParameters.d_SharedParameters, KernelParameters.NumberOfSharedParameters*sizeof(double), cudaMemcpyDeviceToHost) );
}

double ProblemSolver::SingleGetHost(int ProblemNumber, VariableSelection ActualVariable, int SerialNumber)
{
	int idx = ProblemNumber + SerialNumber*KernelParameters.NumberOfThreads;
	
	double Value;
	switch (ActualVariable)
	{
		case TimeDomain:        Value = h_TimeDomain[idx];        break;
		case ActualState:       Value = h_ActualState[idx];       break;
		case ControlParameters: Value = h_ControlParameters[idx]; break;
		case Accessories:       Value = h_Accessories[idx];       break;
	}
	
	return Value;
}

double ProblemSolver::SharedGetHost(int SerialNumber)
{
	double Value = h_SharedParameters[SerialNumber];
	
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
	}
		
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	int idx;
	for (int i=0; i<NumberOfRows; i++)
	{
		for (int j=0; j<NumberOfColumns; j++)
		{
			idx = i + j*NumberOfRows;
			DataFile.width(Width); DataFile << PointerToActualData[idx] << ',';
		}
		DataFile << '\n';
	}
}

void ProblemSolver::Solve(const SolverConfiguration& Configuration)
{
	int GridSize = KernelParameters.NumberOfThreads/Configuration.BlockSize + (KernelParameters.NumberOfThreads%Configuration.BlockSize == 0 ? 0:1);
	
	gpuErrCHK( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
	gpuErrCHK( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
	
	KernelParameters.InitialTimeStep = Configuration.InitialTimeStep;
	
	if ( Configuration.Solver==RKCK45 )
	{
		size_t DynamicSharedMemoryInBytes = 2*KernelParameters.SystemDimension*sizeof(double) + \
											  KernelParameters.NumberOfEvents*sizeof(int) + \
											  KernelParameters.NumberOfEvents*sizeof(double) + \
											  KernelParameters.NumberOfEvents*sizeof(int) + \
											  KernelParameters.NumberOfSharedParameters*sizeof(double);
		
		ParametricODE_Solver_RKCK45<<<GridSize, Configuration.BlockSize, DynamicSharedMemoryInBytes>>> (KernelParameters);
		gpuErrCHK( cudaDeviceSynchronize() );
	}
	
	if ( Configuration.Solver==RKCK45_EH0 )
	{
		size_t DynamicSharedMemoryInBytes = 2*KernelParameters.SystemDimension*sizeof(double) + \
											  KernelParameters.NumberOfSharedParameters*sizeof(double);
		
		ParametricODE_Solver_RKCK45_EH0<<<GridSize, Configuration.BlockSize, DynamicSharedMemoryInBytes>>> (KernelParameters);
		gpuErrCHK( cudaDeviceSynchronize() );
	}
	
	if ( Configuration.Solver==RK4 )
	{
		size_t DynamicSharedMemoryInBytes = KernelParameters.NumberOfEvents*sizeof(int) + \
											KernelParameters.NumberOfEvents*sizeof(double) + \
											KernelParameters.NumberOfEvents*sizeof(int) + \
											KernelParameters.NumberOfSharedParameters*sizeof(double);
		
		ParametricODE_Solver_RK4<<<GridSize, Configuration.BlockSize, DynamicSharedMemoryInBytes>>> (KernelParameters);
		gpuErrCHK( cudaDeviceSynchronize() );
	}
	
	if ( Configuration.Solver==RK4_EH0 )
	{
		size_t DynamicSharedMemoryInBytes = KernelParameters.NumberOfSharedParameters*sizeof(double);
		
		ParametricODE_Solver_RK4_EH0<<<GridSize, Configuration.BlockSize, DynamicSharedMemoryInBytes>>> (KernelParameters);
		gpuErrCHK( cudaDeviceSynchronize() );
	}
	
	gpuErrCHK( cudaMemcpy(h_TimeDomain,  KernelParameters.d_TimeDomain,                                     2*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) );
	gpuErrCHK( cudaMemcpy(h_ActualState, KernelParameters.d_ActualState,     KernelParameters.SystemDimension*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) );
	gpuErrCHK( cudaMemcpy(h_Accessories, KernelParameters.d_Accessories, KernelParameters.NumberOfAccessories*KernelParameters.NumberOfThreads*sizeof(double), cudaMemcpyDeviceToHost) );
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
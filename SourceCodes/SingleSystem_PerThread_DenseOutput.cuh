#ifndef SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H
#define SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H


template <int NT, int SD, int NDO, class Precision>
__forceinline__ __device__ void PerThread_StoreDenseOutput(\
			int        tid, \
			int        r_UpdateDenseOutput, \
			int        r_UpdateStep, \
			int&       r_DenseOutputIndex, \
			Precision* d_DenseOutputTimeInstances, \
			Precision  r_ActualTime, \
			Precision* d_DenseOutputStates, \
			Precision* r_ActualState, \
			int&       r_NumberOfSkippedStores, \
			Precision& r_DenseOutputActualTime, \
			Precision  DenseOutputMinimumTimeStep, \
			Precision  UpperTimeDomain)
{
	if ( r_UpdateDenseOutput == 1 )
	{
		d_DenseOutputTimeInstances[tid + r_DenseOutputIndex*NT] = r_ActualTime;
		
		int DenseOutputStateIndex = tid + r_DenseOutputIndex*NT*SD;
		for (int i=0; i<SD; i++)
		{
			d_DenseOutputStates[DenseOutputStateIndex] = r_ActualState[i];
			DenseOutputStateIndex += NT;
		}
		
		r_DenseOutputIndex++;
		r_NumberOfSkippedStores = 0;
		r_DenseOutputActualTime = MPGOS::FMIN(r_ActualTime+DenseOutputMinimumTimeStep, UpperTimeDomain);
	}
	
	if ( ( r_UpdateDenseOutput == 0 ) && ( r_UpdateStep == 1 ) )
		r_NumberOfSkippedStores++;
}







/*

// ----------------------------------------------------------------------------
template <int NDO>
__forceinline__ __device__ void StoreDenseOutput(IntegratorInternalVariables KernelParameters, int tid, double* ActualState, double ActualTime, double UpperTimeDomain, \
                                                 int& DenseOutputIndex, bool UpdateDenseOutput, double& NextDenseOutputTime)
{
	if ( ( UpdateDenseOutput == 1 ) && ( DenseOutputIndex < KernelParameters.DenseOutputNumberOfPoints ) )
	{
		KernelParameters.d_DenseOutputIndex[tid] = DenseOutputIndex;
		
		int DenseOutputTimeIndex = tid + DenseOutputIndex*KernelParameters.NumberOfThreads;
		KernelParameters.d_DenseOutputTimeInstances[DenseOutputTimeIndex] = ActualTime;
		
		int DenseOutputStateIndex = tid + DenseOutputIndex*KernelParameters.NumberOfThreads*KernelParameters.SystemDimension;
		int TemporaryIndex        = tid;
		for (int i=0; i<KernelParameters.SystemDimension; i++)
		{
			KernelParameters.d_DenseOutputStates[DenseOutputStateIndex] = ActualState[i];
			
			DenseOutputStateIndex += KernelParameters.NumberOfThreads;
			TemporaryIndex += KernelParameters.NumberOfThreads;
		}
		
		DenseOutputIndex++;
		NextDenseOutputTime += KernelParameters.DenseOutputTimeStep;
		
		NextDenseOutputTime = fmin(NextDenseOutputTime, UpperTimeDomain);
	}
}

// ----------
template <>
__forceinline__ __device__ void StoreDenseOutput<0>(IntegratorInternalVariables KernelParameters, int tid, double* ActualState, double ActualTime, double UpperTimeDomain, \
                                                    int& DenseOutputIndex, bool UpdateDenseOutput, double& NextDenseOutputTime)
{}



// ----------------------------------------------------------------------------
template <int NDO>
__forceinline__ __device__ void DenseOutputTimeStepCorrection(IntegratorInternalVariables KernelParameters, int tid, bool& UpdateDenseOutput, int DenseOutputIndex, \
                                                              double NextDenseOutputTime, double ActualTime, double& TimeStep)
{
	if ( KernelParameters.DenseOutputTimeStep <= 0.0 )
	{
		UpdateDenseOutput = 1;
	}
	else if ( DenseOutputIndex < KernelParameters.DenseOutputNumberOfPoints )
	{
		double RequiredDenseOutputTimeStep = NextDenseOutputTime - ActualTime;
		if ( RequiredDenseOutputTimeStep < ( TimeStep + KernelParameters.MinimumTimeStep*1.01 ) )
		{
			TimeStep = fmax(KernelParameters.MinimumTimeStep, RequiredDenseOutputTimeStep);
			UpdateDenseOutput = 1;
		}
	}
}

// ----------
template <>
__forceinline__ __device__ void DenseOutputTimeStepCorrection<0>(IntegratorInternalVariables KernelParameters, int tid, bool& UpdateDenseOutput, int DenseOutputIndex, \
                                                                 double NextDenseOutputTime, double ActualTime, double& TimeStep)
{}*/

#endif
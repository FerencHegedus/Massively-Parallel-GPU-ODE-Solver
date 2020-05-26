#ifndef SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H
#define SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H


template <int NT, int SD, int NDO, class Precision>
__forceinline__ __device__ void PerThread_StoreDenseOutput(\
			int        tid, \
			int        r_UpdateDenseOutput, \
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
	
	if ( r_UpdateDenseOutput == 0 )
		r_NumberOfSkippedStores++;
}


template <int NDO, class Precision>
__forceinline__ __device__ void PerThread_DenseOutputStorageCondition(\
			Precision r_ActualTime, \
			Precision r_DenseOutputActualTime, \
			int       r_DenseOutputIndex, \
			int       r_NumberOfSkippedStores, \
			int       r_EndTimeDomainReached, \
			int       r_UserDefinedTermination, \
			int&      r_UpdateDenseOutput, \
			Struct_SolverOptions<Precision> SolverOptions)
{
	if ( ( r_DenseOutputIndex < NDO ) && ( r_DenseOutputActualTime < r_ActualTime ) && ( r_NumberOfSkippedStores >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
		r_UpdateDenseOutput = 1;
	else
		r_UpdateDenseOutput = 0;
	
	if ( ( r_DenseOutputIndex < NDO ) && ( ( r_EndTimeDomainReached == 1 ) || ( r_UserDefinedTermination == 1 ) ) )
		r_UpdateDenseOutput = 1;
}

#endif
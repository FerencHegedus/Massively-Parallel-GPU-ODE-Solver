#ifndef SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H
#define SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H



__forceinline__ __device__ void PerThread_StoreDenseOutput(\
			int        tid, \
			RegisterStruct &r, \
			__MPGOS_PERTHREAD_PRECISION* d_DenseOutputTimeInstances, \
			__MPGOS_PERTHREAD_PRECISION* d_DenseOutputStates, \
			__MPGOS_PERTHREAD_PRECISION  DenseOutputMinimumTimeStep)
{
	if ( r.UpdateDenseOutput == 1 )
	{
		d_DenseOutputTimeInstances[tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT] = r.ActualTime;

		int DenseOutputStateIndex = tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT*__MPGOS_PERTHREAD_SD;
		for (int i=0; i<__MPGOS_PERTHREAD_SD; i++)
		{
			d_DenseOutputStates[DenseOutputStateIndex] = r.ActualState[i];
			DenseOutputStateIndex += __MPGOS_PERTHREAD_NT;
		}

		r.DenseOutputIndex++;
		r.NumberOfSkippedStores = 0;
		r.DenseOutputActualTime = MPGOS::FMIN(r.ActualTime+DenseOutputMinimumTimeStep, r.TimeDomain[1]);
	}

	if ( r.UpdateDenseOutput == 0 )
		r.NumberOfSkippedStores++;
}


__forceinline__ __device__ void PerThread_DenseOutputStorageCondition(\
			RegisterStruct &r, \
			Struct_SolverOptions<__MPGOS_PERTHREAD_PRECISION> SolverOptions)
{
	if ( ( r.DenseOutputIndex < __MPGOS_PERTHREAD_NDO ) && ( r.DenseOutputActualTime < r.ActualTime ) && ( r.NumberOfSkippedStores >= (SolverOptions.DenseOutputSaveFrequency-1) ) )
		r.UpdateDenseOutput = 1;
	else
		r.UpdateDenseOutput = 0;

	if ( ( r.DenseOutputIndex < __MPGOS_PERTHREAD_NDO ) && ( ( r.EndTimeDomainReached == 1 ) || ( r.UserDefinedTermination == 1 ) ) )
		r.UpdateDenseOutput = 1;
}

#endif

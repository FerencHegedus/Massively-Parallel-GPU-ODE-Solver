#ifndef SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H
#define SINGLESYSTEM_PERTHREAD_DENSEOUTPUT_H


#if __MPGOS_PERTHREAD_NDO > 0


__forceinline__ __device__ void PerThread_HermiteInterpolation(__MPGOS_PERTHREAD_PRECISION &x, __MPGOS_PERTHREAD_PRECISION theta, __MPGOS_PERTHREAD_PRECISION thetaM1, __MPGOS_PERTHREAD_PRECISION dt, __MPGOS_PERTHREAD_PRECISION xb, __MPGOS_PERTHREAD_PRECISION xn, __MPGOS_PERTHREAD_PRECISION xdb, __MPGOS_PERTHREAD_PRECISION xdn)
{
	x = -thetaM1*xb + theta*(xn + thetaM1*((1.0-2.0*theta)*(xn-xb)+dt*(thetaM1*xdb + theta*xdn)));
}

__forceinline__ __device__ void PerThread_SystemToDense(__MPGOS_PERTHREAD_PRECISION * SystemVariables, __MPGOS_PERTHREAD_PRECISION * DenseVariables, SharedStruct s)
{
	for (size_t i = 0; i < __MPGOS_PERTHREAD_DOD; i++)
	{
		DenseVariables[i] = SystemVariables[s.DenseToSystemIndex[i]];
	}
}

__forceinline__ __device__ void PerThread_StoreDenseOutput(\
			int        tid, \
			RegisterStruct &r, \
			SharedStruct &s, \
			__MPGOS_PERTHREAD_PRECISION* d_DenseOutputTimeInstances, \
			__MPGOS_PERTHREAD_PRECISION* d_DenseOutputStates, \
			__MPGOS_PERTHREAD_PRECISION* d_DenseOutputDerivatives, \
			__MPGOS_PERTHREAD_PRECISION  DenseOutputTimeStep)
{
	//printf("%lf\n",DenseOutputTimeStep);
	if ( r.UpdateDenseOutput == 1 ) //dense output without interpolation, store for DDE
	{

		//if(tid==0)
			//printf("idx=%d t=%lf x=%lf xd=%lf\n",r.DenseOutputIndex,r.ActualTime,r.NextDenseState[0],r.NextDerivative[0]);



		//save time
		d_DenseOutputTimeInstances[tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT] = r.ActualTime;

		//save data
		int DenseOutputStateIndex = tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT*__MPGOS_PERTHREAD_DOD;
		for (int i=0; i<__MPGOS_PERTHREAD_DOD; i++)
		{
			d_DenseOutputStates[DenseOutputStateIndex] = r.NextDenseState[i];
			DenseOutputStateIndex += __MPGOS_PERTHREAD_NT;
		}

		#if __MPGOS_PERTHREAD_SAVEDERIVATIVES
			int DenseOutputDerivativeIndex = tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT*__MPGOS_PERTHREAD_DOD;
			for (int i=0; i<__MPGOS_PERTHREAD_DOD; i++)
			{
				d_DenseOutputDerivatives[DenseOutputDerivativeIndex] = r.NextDerivative[i];
				DenseOutputDerivativeIndex += __MPGOS_PERTHREAD_NT;
			}
		#endif

		r.DenseOutputIndex++;
	}

	#if __MPGOS_PERTHREAD_INTERPOLATION
	if ( r.UpdateDenseOutput == 2 )
	{
		while(r.DenseOutputActualTime <= r.ActualTime && r.DenseOutputIndex <__MPGOS_PERTHREAD_NDO)
		{
			//save time
			d_DenseOutputTimeInstances[tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT] = r.DenseOutputActualTime;

			//save data
			int DenseOutputStateIndex = tid + r.DenseOutputIndex*__MPGOS_PERTHREAD_NT*__MPGOS_PERTHREAD_DOD;

			__MPGOS_PERTHREAD_PRECISION theta = (r.DenseOutputActualTime-(r.ActualTime-r.TimeStep))/r.TimeStep;
			__MPGOS_PERTHREAD_PRECISION thetaM1 = theta - 1;

			/*if(tid == 5)
			{
				printf("%d  theta=%lf\t dt=%lf\t t=%lf\t tb=%lf\t tn=%lf\t xb=%lf\t xn=%lf\t xdb=%lf\t xdn=%lf\n", \
					i,theta,r.TimeStep,r.DenseOutputActualTime,r.ActualTime-r.TimeStep,r.ActualTime, \
					r.ActualDenseState[0],r.NextDenseState[0], \
					r.ActualDerivative[0],r.NextDerivative[0]);
			}*/

			for (int i=0; i<__MPGOS_PERTHREAD_DOD; i++)
			{
				PerThread_HermiteInterpolation(d_DenseOutputStates[DenseOutputStateIndex],theta,thetaM1,r.TimeStep,r.ActualDenseState[i],r.NextDenseState[i],r.ActualDerivative[i],r.NextDerivative[i]);
				DenseOutputStateIndex += __MPGOS_PERTHREAD_NT;
			}

			r.DenseOutputIndex++;
			if(r.DenseOutputActualTime == r.TimeDomain[1]) //already on the end of time domain, break loop
			{
				break;
			}
			r.DenseOutputActualTime = MPGOS::FMIN(r.DenseOutputActualTime+DenseOutputTimeStep, r.TimeDomain[1]);
		}
	}


	#endif
}


__forceinline__ __device__ void PerThread_DenseOutputStorageCondition(\
			RegisterStruct &r, \
			SharedStruct &s, \
			Struct_SolverOptions SolverOptions)
{
	//reset dense output counter
	#if __MPGOS_PERTHREAD_ALGORITHM == 2
		if(r.DenseOutputIndex >= __MPGOS_PERTHREAD_NDO)
		{
			r.DenseOutputIndex = 0;
		}
	#endif


	if(SolverOptions.DenseOutputTimeStep <= 0 &&  r.DenseOutputIndex < __MPGOS_PERTHREAD_NDO) //dense output stepped over
	{
		r.UpdateDenseOutput = 1;
		return;
	}

	#if __MPGOS_PERTHREAD_INTERPOLATION
	/*if(threadIdx.x==0 && blockIdx.x==0)
	{
		printf("td=%lf    t=%lf\n",r.DenseOutputActualTime,r.ActualTime);
	}*/
	if(r.DenseOutputActualTime <= r.ActualTime &&  r.DenseOutputIndex < __MPGOS_PERTHREAD_NDO) //dense output stepped over
	{
		r.UpdateDenseOutput = 2;
		return;
	}
	#endif

	r.UpdateDenseOutput = 0;

}

#if __MPGOS_PERTHREAD_ALGORITHM == 2
__forceinline__ __device__ void PerThread_CalculateDelayToTime(int tid, __MPGOS_PERTHREAD_PRECISION T, RegisterStruct &r, SharedStruct SharedSettings, Struct_GlobalVariables GlobalVariables)
{
	for (int i = 0; i < __MPGOS_PERTHREAD_NDELAY; i++)
	{
		__MPGOS_PERTHREAD_PRECISION Tneed = T - SharedSettings.DelayTime[i];
		int Iterations = 0;

		//calculate how many increments are necessary for the i. DelayMemoryIndex
		while(GlobalVariables.d_DenseOutputTimeInstances[tid + __MPGOS_PERTHREAD_NT*r.DelayMemoryIndex[i]] <= Tneed)
		{
			r.DelayMemoryIndex[i]++;
			if(r.DelayMemoryIndex[i] == __MPGOS_PERTHREAD_NDO) //reset delay counter
			{
				r.DelayMemoryIndex[i] = 0;
			}
			Iterations++;
		}

		//update data if necessary
		if(Iterations > 0) //data must be up updated
		{
			int DenseIndex = SharedSettings.DelayToDenseIndex[i];
			int GlobalMemoryStateIdx = tid + r.DelayMemoryIndex[i]*__MPGOS_PERTHREAD_DOD*__MPGOS_PERTHREAD_NT+DenseIndex*__MPGOS_PERTHREAD_NT;
			int GlobalMemoryTimeIdx = tid + __MPGOS_PERTHREAD_NT*r.DelayMemoryIndex[i];

			if(Iterations == 1) //data can be pushed
			{
				r.StateBeforeDelay[i] = r.StateAfterDelay[i];
				r.DerivativeBeforeDelay[i] = r.DerivativeAfterDelay[i];
				r.TimeBeforeDelay[i] = r.TimeAfterDelay[i];

				r.TimeAfterDelay[i] = 	GlobalVariables.d_DenseOutputTimeInstances[GlobalMemoryTimeIdx];
				r.StateAfterDelay[i] = 	GlobalVariables.d_DenseOutputStates[GlobalMemoryStateIdx];
				r.DerivativeAfterDelay[i] = 	GlobalVariables.d_DenseOutputDerivatives[GlobalMemoryStateIdx];

				r.PerStepSizeByDelay[i] = 1.0/(r.TimeAfterDelay[i]-r.TimeBeforeDelay[i]);
			}
			else //full reload from global memory
			{
				r.TimeAfterDelay[i] = 	GlobalVariables.d_DenseOutputTimeInstances[GlobalMemoryTimeIdx];
				r.StateAfterDelay[i] = 	GlobalVariables.d_DenseOutputStates[GlobalMemoryStateIdx];
				r.DerivativeAfterDelay[i] = 	GlobalVariables.d_DenseOutputDerivatives[GlobalMemoryStateIdx];

				//previous step
				if(r.DelayMemoryIndex == 0) //special case - previous step is at __MPGOS_PERTHREAD_NDO-1
				{
					GlobalMemoryTimeIdx =  tid + (__MPGOS_PERTHREAD_NDO-1)*__MPGOS_PERTHREAD_NT;
					GlobalMemoryStateIdx = tid + (__MPGOS_PERTHREAD_NDO-1)*__MPGOS_PERTHREAD_DOD*__MPGOS_PERTHREAD_NT+DenseIndex*__MPGOS_PERTHREAD_NT;
				}
				else
				{
					GlobalMemoryTimeIdx -= __MPGOS_PERTHREAD_NT;
					GlobalMemoryStateIdx -= __MPGOS_PERTHREAD_DOD*__MPGOS_PERTHREAD_NT;
				}
				r.TimeBeforeDelay[i] = 	GlobalVariables.d_DenseOutputTimeInstances[GlobalMemoryTimeIdx];
				r.StateBeforeDelay[i] = 	GlobalVariables.d_DenseOutputStates[GlobalMemoryStateIdx];
				r.DerivativeBeforeDelay[i] = 	GlobalVariables.d_DenseOutputDerivatives[GlobalMemoryStateIdx];


				r.PerStepSizeByDelay[i] = 1.0/(r.TimeAfterDelay[i]-r.TimeBeforeDelay[i]);
			}
		}

		//interpolate
		__MPGOS_PERTHREAD_PRECISION Theta = (Tneed - r.TimeBeforeDelay[i])*r.PerStepSizeByDelay[i];
		__MPGOS_PERTHREAD_PRECISION ThetaM1 = Theta - 1;

		//if(tid==0)
				//printf("i=%d    idx=%d   tneed=%lf    tb=%lf    tn=%lf   pdt=%lf   theta=%lf    xb=%lf    xn=%lf\n",i,r.DelayMemoryIndex[i],Tneed,r.TimeBeforeDelay[i],r.TimeAfterDelay[i],r.PerStepSizeByDelay[i],Theta,r.StateBeforeDelay[i],r.StateAfterDelay[i]);


		PerThread_HermiteInterpolation(r.DelayedState[i],Theta,ThetaM1,r.TimeAfterDelay[i]-r.TimeBeforeDelay[i], \
									r.StateBeforeDelay[i],r.StateAfterDelay[i], \
									r.DerivativeBeforeDelay[i],r.DerivativeAfterDelay[i]);
	}
}
#endif

#endif
#endif

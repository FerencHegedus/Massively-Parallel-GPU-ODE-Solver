#ifndef EPIDEMIC_H
#define EPIDEMIC_H


__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT,
			double*f, double*x, double t, \
			RegisterStruct r, SharedParametersStruct s)
{
	1;
	double NewInfection = s.sp[0] * x[0] *r.xdelay[0]/(1.0 + r.p[0]*r.xdelay[0]);
	//alpha * x(t) * y(t-tau) / (1+beta* y(t-tau))

	double NewRecovery = s.sp[1] * x[1];
	//theta * y(t)

	f[0] = -NewInfection; 							//suspectible
	f[1] = NewInfection - NewRecovery; 	//infected
	f[2] = NewRecovery; 								//recovered
}

// ACCESSORIES
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
	int tid, int NT, \
	RegisterStruct &r, SharedParametersStruct s)
{
	if( r.acc[0] < r.x[1])
	{
		r.acc[0] = r.x[1];
		r.acc[1] = r.t;
	}
}

__forceinline__ __device__ void PerThread_Initialization(\
	int tid, int NT, \
	RegisterStruct &r, SharedParametersStruct s)
{
	r.acc[0] = r.x[1];
	r.acc[0] = r.t;
}

__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, \
			RegisterStruct &r, SharedParametersStruct s)
{

}




#endif

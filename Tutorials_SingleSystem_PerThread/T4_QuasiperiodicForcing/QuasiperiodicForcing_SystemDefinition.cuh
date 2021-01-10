#ifndef T4_PERTHREAD_SYSTEMDEFINITION_H
#define T4_PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
__forceinline__ __device__ void PerThread_OdeFunction(int tid, int NT, \
			double*    F, double*    X, double     T, \
			RegisterStruct r, double * s, int * si)
{
	double rx1 = 1.0/X[0];
	double p   = pow(rx1, r.ControlParameters[10]);

	double s1;
	double c1;
	sincospi(2.0*T, &s1, &c1);

	double s2 = sin(2.0*r.ControlParameters[11]*PI*T+r.ControlParameters[12]);
	double c2 = cos(2.0*r.ControlParameters[11]*PI*T+r.ControlParameters[12]);

	double N;
	double D;
	double rD;

	N = (r.ControlParameters[0]+r.ControlParameters[1]*X[1])*p - r.ControlParameters[2]*(1.0+r.ControlParameters[9]*X[1]) - r.ControlParameters[3]*rx1 - r.ControlParameters[4]*X[1]*rx1 - 1.5*(1.0-r.ControlParameters[9]*X[1]*(1.0/3.0))*X[1]*X[1] - ( r.ControlParameters[5]*s1 + r.ControlParameters[6]*s2 ) * (1.0+r.ControlParameters[9]*X[1]) - X[0]*( r.ControlParameters[7]*c1 + r.ControlParameters[8]*c2 );
	D = X[0] - r.ControlParameters[9]*X[0]*X[1] + r.ControlParameters[4]*r.ControlParameters[9];
	rD = 1.0/D;

	F[0] = X[1];
	F[1] = N*rD;
}

// EVENTS
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, \
			RegisterStruct &r, double * s, int * si)
{
	r.ActualEventValue[0] = r.ActualState[1];
}

__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, \
			RegisterStruct &r, double * s, int * si)
{
	// User defined termination at local maximum
	// IDX is meaningless, as there is only 1 event function
	r.IntegerAccessories[0]++;

	if ( r.IntegerAccessories[0] == 1 )
		r.UserDefinedTermination = 1;
}

// ACCESSORIES
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
	int tid, int NT, \
	RegisterStruct &r, double * s, int * si)
	{
	if ( r.ActualState[0]<r.Accessories[2] )
	{
		r.Accessories[2] = r.ActualState[0];
		r.Accessories[3] = r.ActualTime;
	}
}

__forceinline__ __device__ void PerThread_Initialization(\
	int tid, int NT, \
	RegisterStruct &r, double * s, int * si)
{
	r.Accessories[0] = r.ActualState[0];
	r.Accessories[1] = r.ActualTime;
	r.Accessories[2] = r.ActualState[0];
	r.Accessories[3] = r.ActualTime;

	r.IntegerAccessories[0] = 0; // Event counter for stop condition
}

__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, \
			RegisterStruct &r, double * s, int * si)
{

}

#endif

#ifndef PERTHREAD_SYSTEMDEFINITION_H
#define PERTHREAD_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM

__device__ void PerThread_OdeFunction(int tid, int NT, double* F, double* X, double T, double* cPAR, double* sPAR, double* ACC)
{
	double x1 = X[tid + 0*NT];
	double x2 = X[tid + 1*NT];
	
	double p0  = cPAR[tid +  0*NT];
	double p1  = cPAR[tid +  1*NT];
	double p2  = cPAR[tid +  2*NT];
	double p3  = cPAR[tid +  3*NT];
	double p4  = cPAR[tid +  4*NT];
	double p5  = cPAR[tid +  5*NT];
	double p6  = cPAR[tid +  6*NT];
	double p7  = cPAR[tid +  7*NT];
	double p8  = cPAR[tid +  8*NT];
	double p9  = cPAR[tid +  9*NT];
	double p10 = cPAR[tid + 10*NT];
	double p11 = cPAR[tid + 11*NT];
	double p12 = cPAR[tid + 12*NT];
	
	
	double rx1 = 1.0/x1;
	double p   = pow(rx1, p10);
	
	double s1;
	double c1;
	sincospi(2.0*T, &s1, &c1);
	
	double s2 = sin(2.0*p11*PI*T+p12);
	double c2 = cos(2.0*p11*PI*T+p12);
	
	double N;
	double D;
	double rD;
	
	N = (p0+p1*x2)*p - p2*(1.0+p9*x2) - p3*rx1 - p4*x2*rx1 - 1.5*(1.0-p9*x2/3.0)*x2*x2 - ( p5*s1 + p6*s2 ) * (1.0+p9*x2) - x1*( p7*c1 + p8*c2 );
	D = x1 - p9*x1*x2 + p4*p9;
	rD = 1.0/D;
	
	
	F[tid + 0*NT] = x2;
	F[tid + 1*NT] = N*rD;
}

__device__ void PerThread_OdeProperties(double* RelativeTolerance, double* AbsoluteTolerance, double& MaximumTimeStep, double& MinimumTimeStep, double& TimeStepGrowLimit, double& TimeStepShrinkLimit)
{
	RelativeTolerance[0] = 1e-10;
	RelativeTolerance[1] = 1e-10;
	
	AbsoluteTolerance[0] = 1e-10;
	AbsoluteTolerance[1] = 1e-10;
	
	MaximumTimeStep     = 1.0e6;
	MinimumTimeStep     = 2.0e-12;
	TimeStepGrowLimit   = 5.0;
	TimeStepShrinkLimit = 0.1;
}

// EVENTS

__device__ void PerThread_EventFunction(int tid, int NT, double* EF, double* X, double T, double* cPAR, double* sPAR, double* ACC)
{
	double x2 = X[tid + 1*NT];
	
	EF[tid + 0*NT] = x2;
}

__device__ void PerThread_EventProperties(int* EventDirection, double* EventTolerance, int* EventStopCounter, int& MaxStepInsideEvent)
{
	EventDirection[0]   = -1;
	EventTolerance[0]   = 1e-6;
	EventStopCounter[0] = 1;
		
	MaxStepInsideEvent  = 50;
}

__device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int CNT, double &T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	
}

// ACCESSORIES

__device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	double x1 = X[tid + 0*NT];
	
	if ( x1<ACC[tid + 2*NT] )
	{
		ACC[tid + 2*NT] = x1;
		ACC[tid + 3*NT] = T;
	}
}

__device__ void PerThread_Initialization(int tid, int NT, double T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	double x1 = X[tid + 0*NT];
		
	ACC[tid + 0*NT] = x1;
	ACC[tid + 1*NT] = T;
	ACC[tid + 2*NT] = x1;
	ACC[tid + 3*NT] = T;
}

__device__ void PerThread_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, double* ACC)
{
	TD[tid + 0*NT] = T;
}

#endif
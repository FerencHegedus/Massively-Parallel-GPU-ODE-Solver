#ifndef MPGOS_OVERLOADED_MATHFUNCTIONS_H
#define MPGOS_OVERLOADED_MATHFUNCTIONS_H

namespace MPGOS
{
	// Floating point absolute value
	__forceinline__ __device__ float FABS(float a)
	{
		return fabsf(a);
	}
	
	__forceinline__ __device__ double FABS(double a)
	{
		return fabs(a);
	}
	
	
	// Floating point maximum -------------------------------------------------
	__forceinline__ __device__ float FMAX(float a, float b)
	{
		return fmaxf(a, b);
	}
	
	__forceinline__ __device__ double FMAX(double a, double b)
	{
		return fmax(a, b);
	}
	
	
	// Floating point minimum -------------------------------------------------
	__forceinline__ __device__ float FMIN(float a, float b)
	{
		return fminf(a, b);
	}
	
	__forceinline__ __device__ double FMIN(double a, double b)
	{
		return fmin(a, b);
	}
	
	// Floating point atomic minimum ------------------------------------------
	__forceinline__ __device__ float atomicFMIN(float* address, float val)
	{
		int ret = __float_as_int(*address);
		while ( val < __int_as_float(ret) )
		{
			int old = ret;
			if ( ( ret = atomicCAS((int *)address, old, __float_as_int(val)) ) == old )
				break;
		}
		return __int_as_float(ret);
	}
	
	__forceinline__ __device__ double atomicFMIN(double *address, double val)
	{
		unsigned long long ret = __double_as_longlong(*address);
		while ( val < __longlong_as_double(ret) )
		{
			unsigned long long old = ret;
			if ( ( ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val)) ) == old )
				break;
		}
		return __longlong_as_double(ret);
	}
	
	// Floating point atomic maximum ------------------------------------------
	__forceinline__ __device__ float atomicFMAX(float *address, float val)
	{
		int ret = __float_as_int(*address);
		while ( val > __int_as_float(ret) )
		{
			int old = ret;
			if ( (ret = atomicCAS((int *)address, old, __float_as_int(val)) ) == old )
				break;
		}
		return __int_as_float(ret);
	}
	
	__forceinline__ __device__ double atomicFMAX(double *address, double val)
	{
		unsigned long long ret = __double_as_longlong(*address);
		while ( val > __longlong_as_double(ret) )
		{
			unsigned long long old = ret;
			if ( (ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val)) ) == old )
				break;
		}
		return __longlong_as_double(ret);
	}
}

#endif
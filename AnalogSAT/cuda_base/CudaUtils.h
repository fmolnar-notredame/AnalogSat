//This file is part of AnalogSAT
//Copyright(C) 2019 Ferenc Molnar
//
//AnalogSAT is free software: you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program. If not, see <http://www.gnu.org/licenses/>.


#ifndef ANALOGSAT_CUDA_UTILS_H
#define ANALOGSAT_CUDA_UTILS_H

//#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdexcept>

#define ANALOGSAT_PI 3.141592653589793
#define ANALOGSAT_PI_OVER_2 1.570796326794897

#define SAFEDEL(x) if (x != 0) { cudaFree(x); x = 0; }

//help VisualStudio intellisense to not parse kernel config "<<<>>>" badly
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define MAX_BLOCK_DIM_SIZE 65536

//HACK to make intellisense accept device intrinsics and specific device functions
#ifdef __INTELLISENSE__
#define __launch_bounds__(a,b)
void __syncthreads(void);
void __threadfence(void);
unsigned long long atomicCAS(unsigned long long* ptr, unsigned long long old, unsigned long long newval);
unsigned int atomicCAS(unsigned int* ptr, unsigned int old, unsigned int newval);
unsigned long long __double_as_longlong(double a);
double __longlong_as_double(unsigned long long a);
unsigned int __float_as_uint(float a);
float __uint_as_float(unsigned int a);
#define __CUDACC__
#define __CUDA_ARCH__ 100
#endif

//atomic extensions for pre-pascal arch
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
	__device__ __inline__ double atomicAdd(double* address, double val)
	{
		unsigned long long int* address_as_ull =
			(unsigned long long int*)address;

		unsigned long long int old = *address_as_ull, assumed;
		do
		{
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);
		return __longlong_as_double(old);
	}
#endif

namespace analogsat
{
	void CudaSafe(cudaError_t result, const char* message = NULL);
	void CurandSafe(curandStatus_t result, const char* message = NULL);

	template <class T>
	inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
	{
		CUresult error_result = cuDeviceGetAttribute(attribute, device_attribute, device);
		if (error_result != CUDA_SUCCESS)
		{
			throw std::runtime_error("cuDeviceGetAttribute Failed");
		}
	}


	//arithmetic inlines for GPU only
#ifdef __CUDACC__

	template<typename T>
	__device__ __inline__ T devmax(const T& value1, const T& value2) { return value1 > value2 ? value1 : value2; }

	template<typename T>
	__device__ __inline__ T devabs(const T& x) { return x < 0 ? -x : x; }

	__device__ __inline__ int devand(const int& value1, const int& value2) { return (bool)value1 && (bool)value2; }


	//functors for GPU reduction functions -- this way the reduction kernel can be called with any operator!!!
	template<class T>
	struct functor_devmax
	{
		__device__ T operator() (const T& x, const T& y) { return devmax<T>(x, y); }
	};
	
	template<class T>
	struct functor_devsum
	{
		__device__ T operator() (const T& x, const T& y) { return x + y; }
	};

	struct functor_devand
	{
		__device__ int operator() (const int& x, const int& y) { return devand(x, y); }
	};
	

	//implement atomic max functions for floating-point numbers
	__device__ __inline__ double atomicMax(double* address, double val)
	{
		unsigned long long int* address_as_ull = (unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		do
		{
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(devmax<double>(val, __longlong_as_double(assumed))));
		} while (assumed != old);
		return __longlong_as_double(old);
	}

	__device__ __inline__ float atomicMax(float* address, float val)
	{
		unsigned int* address_as_ull = (unsigned int*)address;
		unsigned int old = *address_as_ull, assumed;
		do
		{
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __float_as_uint(devmax<float>(val, __uint_as_float(assumed))));
		} while (assumed != old);
		return __uint_as_float(old);
	}
#endif

}

#endif

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


#ifndef ANALOGSAT_CUDA_BASE_H
#define ANALOGSAT_CUDA_BASE_H

#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaSatStructs.h"

namespace analogsat
{

	//external shared memory
	extern __shared__ int shmemx[]; 

	//setting the initial condition for type 1 state vectors
	template <typename TFloat>
	__global__ void KernelSetInitial1(int N, int M, TFloat* state);

	//setting the initial condition for type 2 state vectors
	template <typename TFloat>
	__global__ void KernelSetInitial2(int N, int M, TFloat* state);



	//adds mean_am * b * alpha * pi/2 * sin(pi*s[i])  to dxdt[i] for i <= N
	//version 1 solver
	template <typename TFloat>
	__global__ void KernelAdjustState1(CudaSatArgs<TFloat> cons, TFloat* dxdt, TFloat* state);

	//adds mean_am * b * alpha * pi/2 * sin(pi*s[i])  to dxdt[i] for i <= N
	//version 2 solver
	template <typename TFloat>
	__global__ void KernelAdjustState2(CudaSatArgs<TFloat> cons, TFloat* dxdt, TFloat* state);

	//adds mean_am * b * alpha * pi/2 * sin(pi*s[i])  to dxdt[i] for i <= N
	//version 3 solver
	template <typename TFloat>
	__global__ void KernelAdjustState3(CudaSatArgs<TFloat> cons, TFloat* dxdt, TFloat* state);


	//calculate boolean status of each clause, for version 1 state vectors
	template <typename TFloat>
	__global__ void KernelCalculateClauses1(CudaSatArgs<TFloat> cons, TFloat* state, int* boolean, int k);

	//calculate boolean status of each clause, for version 2 state vectors
	template <typename TFloat>
	__global__ void KernelCalculateClauses2(CudaSatArgs<TFloat> cons, TFloat* state, int* boolean, int k, int clauses_per_block);


	//collect method for v3 sat kernels
	template<typename TFloat>
	__global__ void KernelCollect3(int N, TFloat* rhs, TFloat* collect, int* Cn, int* startVar, int* endVar);


	//device inlines to make the Kmi product -- templated all the way, no loops at instruction level
	//partial specialization of templated classes is used
	//final instantiation in the compilation unit where it is used, to allow for inlining
	template<typename TFloat, int K, int Q>
	struct ProdWrapper 	{ static __device__ __forceinline__ TFloat KmiProd(TFloat *s); };

	template<typename TFloat> struct ProdWrapper<TFloat, 1, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat *s) { return (TFloat)1; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 2, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat *s) { return s[1]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 2, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat *s) { return s[0]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 3, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 3, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 3, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 4, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 4, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 4, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 4, 4> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 5, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 5, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 5, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 5, 4> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 5, 5> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 6, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 6, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 6, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 6, 4> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 6, 5> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 6, 6> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 7, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 7, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 7, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 7, 4> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 7, 5> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 7, 6> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 7, 7> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 8, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 4> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 5> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 6> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 7> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[7]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 8, 8> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 9, 1> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 2> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 3> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 4> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 5> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 6> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 7> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[7] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 8> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[8]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 9, 9> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };

	template<typename TFloat> struct ProdWrapper<TFloat, 10, 1 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 2 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 3 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 4 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 5 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 6 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 7 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 8 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[8] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 9 > { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[9]; }; };
	template<typename TFloat> struct ProdWrapper<TFloat, 10, 10> { static __device__ __forceinline__ TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };


	// for each literal the RHS summation is repeated. This loop can be unrolled, but
	// the compiler needs to insert the loop argument into a template argument
	// solution: template metaprogramming, compile-time loop via template argument	
	template<typename TFloat, int K, int Q>
	struct KmiWrapper
	{
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r)
		{
			if (index[Q] != 0)
			{
				TFloat rr = r * ProdWrapper<TFloat, K, Q + 1>::KmiProd(s); // calculate r * Kmi
				rr *= index[Q] < 0 ? (TFloat)-2.0 : (TFloat)2.0;			//multiply by sign
				atomicAdd(&rhs[abs(index[Q])], rr);							//accumulate RHS contribution via atomics
				KmiWrapper<TFloat, K, Q + 1>::KmiFunc(rhs, index, s, r);				//template recursion
			}
		}

		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q)
		{
			if (index[Q] != 0)
			{
				TFloat kmi = ProdWrapper<TFloat, K, Q + 1>::KmiProd(s);
				TFloat rr = r * (kmi < 0 ? -kmi : kmi) * tanh(q * s[Q]); //knorm * am * |Kmi| * tanh(q * s[i])
				rr *= index[Q] < 0 ? (TFloat)-1.0 : (TFloat)1.0;
				atomicAdd(&rhs[abs(index[Q])], rr);
				KmiWrapper<TFloat, K, Q + 1>::KmiFuncTanh(rhs, index, s, r, q);
			}
		}
	};

	// compile-time loop termination
	template<typename TFloat> struct KmiWrapper<TFloat, 1, 1> 
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 2, 2> 
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 3, 3> 
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 4, 4> 
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 5, 5>
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 6, 6>
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 7, 7> 
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 8, 8>
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 9, 9>
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct KmiWrapper<TFloat, 10, 10> 
	{ 
		static __device__ __forceinline__ void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static __device__ __forceinline__ void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
}


#endif

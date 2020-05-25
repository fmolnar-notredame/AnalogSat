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


#include "CudaSatBase.h"

namespace analogsat
{
	//transform a uniform random number to the initial SPIN state
	template<typename TFloat>
	__device__ __forceinline__ TFloat InitStateTransform(TFloat s)
	{
		return s * (TFloat)2 - (TFloat)1;
	}

	//transform a uniform random number to the initial AUXILIARY state
	template<typename TFloat>
	__device__ __forceinline__ TFloat InitAuxTransform(TFloat s)
	{
		return s * (TFloat)1 + (TFloat)1;
	}


	//setting the initial condition for type 1 state vectors
	template <typename TFloat>
	__global__ void KernelSetInitial1(int N, int M, TFloat* state)
	{		
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= 1)
		{
			if (i < N + 1)
			{
				state[i] = InitStateTransform(state[i]);
			}
			else if (i < N + M + 1)
			{				
				state[i] = InitAuxTransform(state[i]);
			}
		}
	}

	//setting the initial condition for type 2 state vectors
	template <typename TFloat>
	__global__ void KernelSetInitial2(int N, int M, TFloat* state)
	{		
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= 2)
		{
			if (i < N + 2)
			{
				state[i] = InitStateTransform(state[i]);				
			}
			else if (i < N + M + 2)
			{
				state[i] = InitAuxTransform(state[i]);
			}
		}
	}

	//adds mean_am * b * alpha * pi/2 * sin(pi*s[i])  to dxdt[i] for i <= N
	//version 1 solver
	template <typename TFloat>
	__global__ void KernelAdjustState1(CudaSatArgs<TFloat> cons, TFloat* dxdt, TFloat* state)
	{		
		TFloat mean_am = *(cons.MEAN_AM) / cons.M;
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i > 0 && i < cons.N + 1)
		{
			TFloat s = state[i];
			TFloat d = mean_am * cons.ALPHA * cons.B * (TFloat)ANALOGSAT_PI_OVER_2 * sin(s * (TFloat)ANALOGSAT_PI);
			dxdt[i] += d;
		}
	}

	//adds mean_am * b * alpha * pi/2 * sin(pi*s[i])  to dxdt[i] for i <= N
	//version 2 solver
	template <typename TFloat>
	__global__ void KernelAdjustState2(CudaSatArgs<TFloat> cons, TFloat* dxdt, TFloat* state)
	{
		TFloat mean_am = *(cons.MEAN_AM) / cons.M;
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= 2 && i < cons.N + 2)
		{
			TFloat s = state[i];
			TFloat d = mean_am * cons.ALPHA * cons.B * (TFloat)ANALOGSAT_PI_OVER_2 * sin(s * (TFloat)ANALOGSAT_PI);
			dxdt[i] += d;
		}
	}

	//adds mean_am * b * alpha * pi/2 * sin(pi*s[i])  to dxdt[i] for i <= N
	//version 3 solver
	template <typename TFloat>
	__global__ void KernelAdjustState3(CudaSatArgs<TFloat> cons, TFloat* dxdt, TFloat* state)
	{		
		TFloat mean_am = cons.MEAN_AM[0] / cons.M; //mean am !
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i >= 2 && i < cons.N + 2)
		{
			TFloat s = state[i];
			TFloat d = mean_am * cons.ALPHA * cons.B * (TFloat)ANALOGSAT_PI_OVER_2 * sin(s * (TFloat)ANALOGSAT_PI);
			dxdt[i] += d;
		}
	}

	//calculate boolean status of each clause, for version 1 state vectors
	template <typename TFloat>
	__global__ void KernelCalculateClauses1(CudaSatArgs<TFloat> cons, TFloat* state, int* boolean, int k)
	{
		bool clause = false;
		int index;
		bool lit;
		int loc;

		int j = blockIdx.x * blockDim.x + threadIdx.x;

		if (j < cons.M)
		{
			for (int q = 0; q < k; q++)
			{
				//load (coalesced)
				index = cons.GC[j + q * cons.STRIDE];

				//extract sign and index, combine with clause bool
				loc = index < 0 ? -index : index;
				lit = state[loc] > (TFloat)0.0; // gather (uncoalesced)
				if (index < 0) lit = !lit; //negate if needed
				clause |= lit;
			}

			//store
			boolean[j] = clause ? 0 : 1; //1 if violated
		}
	}

	//calculate boolean status of each clause, for version 2 state vectors
	template <typename TFloat>
	__global__ void KernelCalculateClauses2(CudaSatArgs<TFloat> cons, TFloat* state, int* boolean, int k, int clauses_per_block)
	{
		//shared memory: k * clauses_per_block of TFloats needed for RHS, the same amount of int needed for clauses
		int* sh_lit = reinterpret_cast<int*>(shmemx);		

		int clause = false;
		int idx, index;

		int i = blockIdx.x * blockDim.x + threadIdx.x;	//index in clause-literals

		//load (coalesced)
		index = cons.GC[i];
		idx = index;

		//transform to get state index
		idx *= idx < 0 ? -1 : 1;

		int lit = state[idx] > 0;			// gather (uncoalesced), convert to bool
		lit = index < 0 ? !(bool)lit : lit;	//negate if needed

		//deposit lit to shared
		sh_lit[threadIdx.x] = lit;
		
		__syncthreads();

		//only the first threads compute the clause reduction (not the best way, but optimize further ONLY if this really becomes a bottleneck)
		if (threadIdx.x < clauses_per_block)  //use one thread per clause
		{
			for (int z = 0; z < k; z++)
			{
				clause |= sh_lit[threadIdx.x * k + z]; //bank conflicts, yea yea..
			}

			int j = blockIdx.x * clauses_per_block + threadIdx.x;
			boolean[j] = clause ? 0 : 1; //1 if violated
		}
	}

	//collect method for v3 sat kernels
	template<typename TFloat>
	__global__ void KernelCollect3(int N, TFloat* rhs, TFloat* collect, int* Cn, int* startVar, int* endVar)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32; //one warp per variable
		int w = threadIdx.x & 0x1f; //lane ID == index in warp

		TFloat total = 0;
		TFloat sum = 0;

		if (i < N + 2) //this is true or false for an entire warp. No divergence.
		{
			int start = startVar[i];	//coalesced load, (broadcast), underfill unless #warps % 8 == 0 (so whole 32-byte transactions are fully utilized)
			int end = endVar[i];

			for (int q = start; q < end; q += 32)
			{
				//load the next lane of indices
				int loc = Cn[q + w]; //coalesced load of the location pointers (into the collect array)

				sum = loc >= 0 ? collect[loc] : 0; // uncoalesced load (!) gather

				//reduction in warp
#if CUDART_VERSION >= 9000
				sum += __shfl_down_sync(0xffffffff, sum, 16, 32);
				sum += __shfl_down_sync(0xffffffff, sum, 8, 32);
				sum += __shfl_down_sync(0xffffffff, sum, 4, 32);
				sum += __shfl_down_sync(0xffffffff, sum, 2, 32);
				sum += __shfl_down_sync(0xffffffff, sum, 1, 32);
#else
				sum += __shfl_down(sum, 16);
				sum += __shfl_down(sum, 8);
				sum += __shfl_down(sum, 4);
				sum += __shfl_down(sum, 2);
				sum += __shfl_down(sum, 1);
#endif

				//first thread in the warp holds the result
				total += sum; //so this makes sense only in the first lane in the warp. For the rest, this is meaningless.
			}

			if (w == 0) //first lane: write the result
				rhs[i] = total; //coalesced write, underfill
		}
	}

	//instantiate
	template __global__ void KernelSetInitial1<float>(int N, int M, float* state);
	template __global__ void KernelSetInitial2<float>(int N, int M, float* state);

	template __global__ void KernelAdjustState1<float>(CudaSatArgs<float> cons, float* dxdt, float* state);
	template __global__ void KernelAdjustState2<float>(CudaSatArgs<float> cons, float* dxdt, float* state);
	template __global__ void KernelAdjustState3<float>(CudaSatArgs<float> cons, float* dxdt, float* state);

	template __global__ void KernelCalculateClauses1<float>(CudaSatArgs<float> cons, float* state, int* boolean, int k);
	template __global__ void KernelCalculateClauses2<float>(CudaSatArgs<float> cons, float* state, int* boolean, int k, int clauses_per_block);

	template __global__ void KernelCollect3<float>(int N, float* rhs, float* collect, int* Cn, int* startVar, int* endVar);



	template __global__ void KernelSetInitial1<double>(int N, int M, double* state);
	template __global__ void KernelSetInitial2<double>(int N, int M, double* state);

	template __global__ void KernelAdjustState1<double>(CudaSatArgs<double> cons, double* dxdt, double* state);
	template __global__ void KernelAdjustState2<double>(CudaSatArgs<double> cons, double* dxdt, double* state);
	template __global__ void KernelAdjustState3<double>(CudaSatArgs<double> cons, double* dxdt, double* state);

	template __global__ void KernelCalculateClauses1<double>(CudaSatArgs<double> cons, double* state, int* boolean, int k);
	template __global__ void KernelCalculateClauses2<double>(CudaSatArgs<double> cons, double* state, int* boolean, int k, int clauses_per_block);

	template __global__ void KernelCollect3<double>(int N, double* rhs, double* collect, int* Cn, int* startVar, int* endVar);

}
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


#include <stdexcept>

#include "CudaSatTanh2.h"
#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaReduce.h"
#include "../util/utils.h"
#include "CudaSatBase.h"

using namespace std;

namespace analogsat
{
	//simplified version, no templates on k or clauses_per_block -> params, and external shared
	template <typename TFloat>
	__global__ void KernelTanhRHSv2(CudaSatArgs<TFloat> cons, TFloat* state, TFloat* rhs, int k, int clauses_per_block, TFloat q)
	{
		TFloat* sh_state = reinterpret_cast<TFloat*>(shmemx);

		int i = blockIdx.x * blockDim.x + threadIdx.x;	//index in clause-literals
		int threadDivK = threadIdx.x / k;				//which clause (out of the clauses processed by this block)
		int threadModK = threadIdx.x - threadDivK * k;	//which literal within the clause

		//grab the desired section of clauses
		int idx = cons.GC[i]; //coalesced load

		//transform to get state index, and keep the floating sign	
		TFloat sign = idx < 0 ? (TFloat)-1.0 : (TFloat)1.0;
		idx *= idx < 0 ? -1 : 1;

		//gather, calculate, and store spin
		TFloat st = state[idx];			//load uncoalesced (!) - arrange clauses/literals to maximize coalescation
		st = (TFloat)1.0 - st * sign;	//spin
		sh_state[threadIdx.x] = st;		//store shared (no bank conflict)

		//sync required, shared access pattern changes
		__syncthreads();

		//go over the clauses that belongs to us, compute km and kmi
		TFloat km = cons.KNORM;
		TFloat kmi = km;


		for (int z = 0; z < k; z++)
		{
			TFloat s = sh_state[threadDivK * k + z]; //no bank conflict, k-way broadcast
			km *= s;
			kmi *= (threadModK == z) ? (TFloat)1.0 : s;
		}

		__syncthreads();

		//load am and deal with am's rhs
		//one thread per CLAUSE can do it, but to make it coalested, it should be done by the first clauses_per_block threads.
		//for this, the first clauses_per_block threads also need km's for those CLAUSES, which are now in the posession of k consecutive threads

		//the 1st thread of each clause deposits their km
		if (threadModK == 0) sh_state[threadDivK] = km * km; //no two threads write the same address, no bank conflicts

		__syncthreads();

		if (threadIdx.x < clauses_per_block)
		{
			//auxiliary rhs
			int offset = blockIdx.x * clauses_per_block + threadIdx.x + cons.N + 2;
			TFloat amm = state[offset];					//load (coalesced)
			rhs[offset] = sh_state[threadIdx.x] * amm;	//load shared (no conflict), store (coalesced)		

			//replace km with am in shared
			sh_state[threadIdx.x] = amm;		//store shared (no conflict)
		}
		__syncthreads();

		//broadcast am's to each thread
		TFloat am = sh_state[threadDivK];

		//calculate and write result		
		TFloat re = abs(kmi) * am * sign * tanh(q * st);

		//write result
		if (idx >= 2) atomicAdd(&rhs[idx], re);

		if (cons.B > (TFloat)0)
		{
			//am is only summable once per clause, so zero out for each thread that is not the leader of a clause
			if (threadModK > 0) am = 0;

#if CUDART_VERSION >= 9000
			//warp-reduce am (non-computed threads will add zero)
			am += __shfl_down_sync(0xffffffff, am, 16, 32);
			am += __shfl_down_sync(0xffffffff, am, 8, 32);
			am += __shfl_down_sync(0xffffffff, am, 4, 32);
			am += __shfl_down_sync(0xffffffff, am, 2, 32);
			am += __shfl_down_sync(0xffffffff, am, 1, 32);
#else
			am += __shfl_down(am, 16);
			am += __shfl_down(am, 8);
			am += __shfl_down(am, 4);
			am += __shfl_down(am, 2);
			am += __shfl_down(am, 1);
#endif

			//deposit the sum to global
			if (threadIdx.x % 32 == 0) atomicAdd(cons.MEAN_AM, am); //first lane in the warp deposits the result
		}
	}

	template <typename TFloat>
	CudaSatTanh2<TFloat>::CudaSatTanh2()
	{
		q = (TFloat)1.1;
	}

	template <typename TFloat>
	CudaSatTanh2<TFloat>::~CudaSatTanh2()
	{
		Free();
	}

	template<typename TFloat>
	TFloat CudaSatTanh2<TFloat>::Get_Q() const { return q; }	//parameter q

	template<typename TFloat>
	void CudaSatTanh2<TFloat>::Set_Q(TFloat _q)  { q = _q; }	//parameter q


	template <typename TFloat>
	void CudaSatTanh2<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		//reset the global mean am
		if (b > (TFloat)0) CudaSafe(cudaMemsetAsync(gMeanAm, 0, sizeof(TFloat)));

		dxdt.SetZero();
		cons.B = b * cons.KNORM * cons.KNORM; // premultiply the bias term with normalization	

		//call the RHS kernel
		int shmem2t = k * clauses_per_block * sizeof(TFloat);
		KernelTanhRHSv2<TFloat> KERNEL_ARGS3(blocks, threads, shmem2t)(cons, state, dxdt, k, clauses_per_block, q);

		if (b > (TFloat)0)
		{
			//add the sine term for each variable
			dim3 blocks2((int)ceil((float)(n + 2) / threads.x));
			KernelAdjustState2<TFloat> KERNEL_ARGS2(blocks2, threads)(cons, dxdt, state);
		}
	}

	//instantiate the class for float and double
	template class CudaSatTanh2<double>;
	template class CudaSatTanh2<float>;
}

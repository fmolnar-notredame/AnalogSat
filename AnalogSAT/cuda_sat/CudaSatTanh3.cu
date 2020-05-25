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


#include <chrono>
#include <stdexcept>

#include "CudaSatTanh3.h"
#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaReduce.h"
#include "../util/utils.h"
#include "CudaSatBase.h"

using namespace std;

namespace analogsat
{
	//one literal per thread ALWAYS,  using shared memory to calculate kmi*km
	//groups of k threads cooperate
	//no underfilled clauses, clauses rounded up to full blocks, so no condition on m is needed
	template <typename TFloat>
	__global__ void KernelTanhRHSv3(CudaSatArgs<TFloat> cons, TFloat* state, TFloat* rhs, TFloat* collect, int k, int clauses_per_block, TFloat q)
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

		//go over the clauses that belongs to us, compute km*kmi
		TFloat km = cons.KNORM;
		TFloat kmi = km;

#pragma unroll
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
		if (threadModK == 0) sh_state[threadDivK] = km * km; //no two threads write the same bank, no bank conflicts

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
		if (idx >= 2) collect[i] = abs(kmi) * am * sign * tanh(q * st); //coalesced
	}

	template <typename TFloat>
	CudaSatTanh3<TFloat>::CudaSatTanh3()
	{
		q = (TFloat)1.1;
	}

	template <typename TFloat>
	CudaSatTanh3<TFloat>::~CudaSatTanh3()
	{
		Free();
	}


	template<typename TFloat>
	TFloat CudaSatTanh3<TFloat>::Get_Q() const { return q; }	//parameter q

	template<typename TFloat>
	void CudaSatTanh3<TFloat>::Set_Q(TFloat _q)  { q = _q; }	//parameter q


	template <typename TFloat>
	void CudaSatTanh3<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		//step 0: reduce sum auxiliary variables (deterministic sum)
		if (b > (TFloat)0) CudaReduceSum<TFloat>(m, state + n + 2, gAux1, gAux2);

		dxdt.SetZero();
		cons.B = b;
		cons.MEAN_AM = gAux1;

		//step 1: calculate RHS terms
		int shmem = k * clauses_per_block * sizeof(TFloat);
		KernelTanhRHSv3<TFloat> KERNEL_ARGS3(blocks, threads, shmem)(cons, state, dxdt, gCollect, k, clauses_per_block, q);

		//step 2: add up the RHS terms for each variable (deterministic sum)
		KernelCollect3<TFloat> KERNEL_ARGS2(blocks_collect, threads_collect)(n, dxdt, gCollect, gCn, gStartVar, gEndVar);

		if (b > (TFloat)0)
		{
			//step 3: add the sine term for each variable (using the deterministic mean from step 0)		
			dim3 blocks2((int)ceil((float)(n + 2) / threads.x));
			KernelAdjustState3<TFloat> KERNEL_ARGS2(blocks2, threads)(cons, dxdt, state);
		}
	}

	//instantiate the class for float and double
	template class CudaSatTanh3<double>;
	template class CudaSatTanh3<float>;
}

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


#include "CudaSatTanh1.h"
#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaReduce.h"
#include "../util/utils.h"
#include "CudaSatBase.h"

using namespace std;

namespace analogsat
{
	template <typename TFloat, int K>
	__global__ void KernelTanhRHSv1(CudaSatArgs<TFloat> cons, TFloat* rhs, const TFloat* __restrict__ state, TFloat q)
	{
		int index[K];
		TFloat s[K];
		TFloat km, am = 0;

		int j = blockIdx.x * blockDim.x + threadIdx.x;
		if (j < cons.M)
		{
			km = cons.KNORM;

#pragma unroll
			for (int z = 0; z < K; z++) index[z] = cons.GC[j + z * cons.STRIDE]; //load (coalesced)

#pragma unroll
			for (int z = 0; z < K; z++)
			{
				s[z] = index[z] != 0 ? state[abs(index[z])] : (TFloat)-1.0;	//gather (uncoalesced), abs(index) is a device-intrinsic
				s[z] = index[z] < 0 ? (TFloat)1.0 + s[z] : (TFloat)1.0 - s[z];
				km *= s[z];
			}

			//auxiliary rhs
			int p = cons.N + j + 1;
			am = state[p];		//load (coalesced)
			rhs[p] = am * km * km;	//store (coalesced)

			//rhs for variables - invoke templated loop
			KmiWrapper<TFloat, K, 0>::KmiFuncTanh(rhs, index, s, cons.KNORM * am, q);
		}

		if (cons.B > (TFloat)0)
		{
			__syncthreads();

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
	CudaSatTanh1<TFloat>::CudaSatTanh1()
	{
		q = (TFloat)1.1;
	}


	template <typename TFloat>
	CudaSatTanh1<TFloat>::~CudaSatTanh1()
	{
		Free();
	}


	template<typename TFloat>
	TFloat CudaSatTanh1<TFloat>::Get_Q() const { return q; }	//parameter q

	template<typename TFloat>
	void CudaSatTanh1<TFloat>::Set_Q(TFloat _q)  { q = _q; }	//parameter q	

	template <typename TFloat>
	void CudaSatTanh1<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		//reset the global mean am
		if (b > (TFloat)0) CudaSafe(cudaMemsetAsync(gMeanAm, 0, sizeof(TFloat)));			

		dxdt.SetZero();
		cons.B = b * cons.KNORM * cons.KNORM; // premultiply the bias term with normalization

		//run the RHS calculation
		switch (k)
		{
		case 0:	throw runtime_error("SAT problem has not been set");
		case 1:	KernelTanhRHSv1<TFloat, 1> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 2:	KernelTanhRHSv1<TFloat, 2> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 3:	KernelTanhRHSv1<TFloat, 3> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 4: KernelTanhRHSv1<TFloat, 4> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 5: KernelTanhRHSv1<TFloat, 5> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 6: KernelTanhRHSv1<TFloat, 6> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 7: KernelTanhRHSv1<TFloat, 7> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 8: KernelTanhRHSv1<TFloat, 8> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 9: KernelTanhRHSv1<TFloat, 9> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		case 10: KernelTanhRHSv1<TFloat, 10> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state, q); break;
		default: throw runtime_error("unsupported problem size");
		}

		if (b > (TFloat)0)
		{
			//compute the additional sine term for each variable
			dim3 blocks2((int)ceil((float)(n + 1) / threads.x));
			KernelAdjustState1<TFloat> KERNEL_ARGS2(blocks2, threads)(cons, dxdt, state);
		}
	}

	//instantiate the class for float and double
	template class CudaSatTanh1<double>;
	template class CudaSatTanh1<float>;
}

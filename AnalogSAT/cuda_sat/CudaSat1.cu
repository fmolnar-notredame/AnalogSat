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


#include "CudaSat1.h"
#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaReduce.h"
#include "../util/utils.h"

#include "CudaSatBase.h"

using namespace std;

// maximum number of literals in a clause supported by this solver
#define MAX_K 10

namespace analogsat
{

	//older implementation, for reference - DO NOT DELETE, KEEP FOR ILLUSTRATION
	/*
	//r.h.s calculation: m clauses to evaluate
	//each thread operates on the clause in the current workwindow	
	// kept for versioning, original implementation
	template <typename TFloat>
	__global__ void KernelCalculateRHSc3(TFloat* rhs, TFloat* state)
	{
		const devconstSat1<TFloat>& cons = GetConsSat1<TFloat>();
		int* clauses = cons.GC;
		TFloat knorm = cons.KNORM;

		int index1 = 0, index2 = 0, index3 = 0;
		TFloat sign1, sign2, sign3;
		TFloat km, am = 0;
		TFloat s1, s2, s3;

		int j = blockIdx.x * blockDim.x + threadIdx.x;
		if (j < cons.M)
		{
			km = knorm;

			//literal 1
			index1 = clauses[j];					//load (coalesced)
			sign1 = index1 < 0 ? (TFloat)-1.0 : (TFloat)1.0f;
			index1 = index1 < 0 ? -index1 : index1;

			//literal 2
			index2 = clauses[j + cons.STRIDE];		//load (coalesced)
			sign2 = index2 < 0 ? (TFloat)-1.0 : (TFloat)1.0;
			index2 = index2 < 0 ? -index2 : index2;

			//literal 3
			index3 = clauses[j + 2 * cons.STRIDE];	//load (coalesced)
			sign3 = index3 < 0 ? (TFloat)-1.0 : (TFloat)1.0;
			index3 = index3 < 0 ? -index3 : index3;


			s1 = state[index1];				//gather (uncoalesced)	
			s1 = (TFloat)1.0 - s1 * sign1;	//spin
			km *= s1;						//multiply km

			s2 = state[index2];				//gather (uncoalesced)	
			s2 = (TFloat)1.0 - s2 * sign2;	//spin
			km *= s2;						//multiply km

			s3 = state[index3];				//gather (uncoalesced)
			s3 = (TFloat)1.0 - s3 * sign3;	//spin
			km *= s3;						//multiply km

			//auxiliary rhs
			int p = cons.N + j + 1;
			am = state[p];		//load (coalesced)
			rhs[p] = am * km;	//store (coalesced)
		}

		//rhs for variables - variable 0 never changes - out of range j threads will default to index=0, nothing happens
		if (index1 > 0)
		{
			TFloat r = (TFloat)2.0 * km * knorm * s2 * s3 * am * sign1;
			atomicAdd(&rhs[index1], r);
		}

		if (index2 > 0)
		{
			TFloat r = (TFloat)2.0 * km * knorm * s1 * s3 * am * sign2;
			atomicAdd(&rhs[index2], r);
		}

		if (index3 > 0)
		{
			TFloat r = (TFloat)2.0 * km * knorm * s1 * s2 * am * sign3;
			atomicAdd(&rhs[index3], r);
		}
	}

	//modified r.h.s calculation: am values are summed across all clauses for later adjustment	
	// kept for versioning, how to warp-sum am values
	template <typename TFloat>
	__global__ void KernelCalculateRHSc3_mod(TFloat* rhs, TFloat* state)
	{
		const devconstSat1<TFloat>& cons = GetConsSat1<TFloat>();
		int* clauses = cons.GC;
		TFloat knorm = cons.KNORM;

		int index1 = 0, index2 = 0, index3 = 0;
		TFloat sign1, sign2, sign3;
		TFloat km, am = 0;
		TFloat s1, s2, s3;

		int j = blockIdx.x * blockDim.x + threadIdx.x;
		if (j < cons.M)
		{
			km = knorm;

			//literal 1
			index1 = clauses[j];					//load (coalesced)
			sign1 = index1 < 0 ? (TFloat)-1.0 : (TFloat)1.0f;
			index1 = index1 < 0 ? -index1 : index1;

			//literal 2
			index2 = clauses[j + cons.STRIDE];		//load (coalesced)
			sign2 = index2 < 0 ? (TFloat)-1.0 : (TFloat)1.0;
			index2 = index2 < 0 ? -index2 : index2;

			//literal 3
			index3 = clauses[j + 2 * cons.STRIDE];	//load (coalesced)
			sign3 = index3 < 0 ? (TFloat)-1.0 : (TFloat)1.0;
			index3 = index3 < 0 ? -index3 : index3;


			s1 = state[index1];				//gather (uncoalesced)	
			s1 = (TFloat)1.0 - s1 * sign1;	//spin
			km *= s1;						//multiply km

			s2 = state[index2];				//gather (uncoalesced)	
			s2 = (TFloat)1.0 - s2 * sign2;	//spin
			km *= s2;						//multiply km

			s3 = state[index3];				//gather (uncoalesced)
			s3 = (TFloat)1.0 - s3 * sign3;	//spin
			km *= s3;						//multiply km

			//auxiliary rhs
			int p = cons.N + j + 1;
			am = state[p];		//load (coalesced)
			rhs[p] = am * km;	//store (coalesced)
		}

		//sum-reduce am values in warp -- out of range j threads will sum zeros, OK
		TFloat sum = am;
		sum += __shfl_down_sync(0xffffffff, sum, 16, 32);
		sum += __shfl_down_sync(0xffffffff, sum, 8, 32);
		sum += __shfl_down_sync(0xffffffff, sum, 4, 32);
		sum += __shfl_down_sync(0xffffffff, sum, 2, 32);
		sum += __shfl_down_sync(0xffffffff, sum, 1, 32);
		
		//deposit the sum to global
		if (threadIdx.x % 32 == 0) atomicAdd(GetMeanAm<TFloat>(), sum); //first lane in the warp deposits the result


		//rhs for variables - variable 0 never changes - out of range j threads will default to index=0, nothing happens
		if (index1 > 0)
		{
			TFloat r = (TFloat)2.0 * km * knorm * s2 * s3 * am * sign1;
			atomicAdd(&rhs[index1], r);
		}

		if (index2 > 0)
		{
			TFloat r = (TFloat)2.0 * km * knorm * s1 * s3 * am * sign2;
			atomicAdd(&rhs[index2], r);
		}

		if (index3 > 0)
		{
			TFloat r = (TFloat)2.0 * km * knorm * s1 * s2 * am * sign3;
			atomicAdd(&rhs[index3], r);
		}
	}

	// kept for versioning, and how to avoid spin variables
	template <typename TFloat>
	__global__ void KernelCalculateRHSc4(TFloat* rhs, TFloat* state)
	{
		const devconstSat1<TFloat>& cons = GetConsSat1<TFloat>();
		int* clauses = cons.GC;
		TFloat knorm = cons.KNORM;

		int index1, index2, index3, index4;
		TFloat km;
		TFloat s1, s2, s3, s4;

		int j = blockIdx.x * blockDim.x + threadIdx.x;
		if (j < cons.M)
		{
			km = knorm;

			index1 = clauses[j];					//load (coalesced)
			index2 = clauses[j + cons.STRIDE];		//load (coalesced)
			index3 = clauses[j + 2 * cons.STRIDE];	//load (coalesced)
			index4 = clauses[j + 3 * cons.STRIDE];	//load (coalesced)

			s1 = index1 != 0 ? state[abs(index1)] : (TFloat)-1.0;				//gather (uncoalesced), abs(index) is a device-intrinsic
			s1 = index1 < 0 ? (TFloat)1.0 + s1 : (TFloat)1.0 - s1;
			km *= s1;

			s2 = index2 != 0 ? state[abs(index2)] : (TFloat)-1.0;				//gather (uncoalesced), abs(index) is a device-intrinsic
			s2 = index2 < 0 ? (TFloat)1.0 + s2 : (TFloat)1.0 - s2;
			km *= s2;

			s3 = index3 != 0 ? state[abs(index3)] : (TFloat)-1.0;				//gather (uncoalesced), abs(index) is a device-intrinsic
			s3 = index3 < 0 ? (TFloat)1.0 + s3 : (TFloat)1.0 - s3;
			km *= s3;

			s4 = index4 != 0 ? state[abs(index4)] : (TFloat)-1.0;				//gather (uncoalesced), abs(index) is a device-intrinsic
			s4 = index4 < 0 ? (TFloat)1.0 + s4 : (TFloat)1.0 - s4;
			km *= s4;


			//auxiliary rhs
			int p = cons.N + j + 1;
			float  am = state[p];		//load (coalesced)
			rhs[p] = am * km;	//store (coalesced)

			//rhs for variables - variable 0 never changes
			if (index1 != 0)
			{
				TFloat r = km * knorm * s2 * s3 * s4 * am;
				r *= index1 < 0 ? (TFloat)-2.0 : (TFloat)2.0;
				atomicAdd(&rhs[abs(index1)], r);
			}

			if (index2 != 0)
			{
				TFloat r = km * knorm * s1 * s3 * s4 * am;
				r *= index2 < 0 ? (TFloat)-2.0 : (TFloat)2.0;
				atomicAdd(&rhs[abs(index2)], r);
			}

			if (index3 != 0)
			{
				TFloat r = km * knorm * s1 * s2 * s4 * am;
				r *= index3 < 0 ? (TFloat)-2.0 : (TFloat)2.0;
				atomicAdd(&rhs[abs(index3)], r);
			}

			if (index4 != 0)
			{
				TFloat r = km * knorm * s1 * s2 * s3 * am;
				r *= index4 < 0 ? (TFloat)-2.0 : (TFloat)2.0;
				atomicAdd(&rhs[abs(index4)], r);
			}
		}
	}
	*/
	
	template <typename TFloat, int K>
	__global__ void KernelCalculateRHSv1(CudaSatArgs<TFloat> cons, TFloat* rhs, const TFloat* __restrict__ state)
	{
		int index[K];
		TFloat s[K];
		TFloat km, am = 0;

		int j = blockIdx.x * blockDim.x + threadIdx.x;
		if (j < cons.M)
		{
			km = cons.KNORM;

			#pragma unroll
			for (int q = 0; q < K; q++) index[q] = cons.GC[j + q * cons.STRIDE]; //load (coalesced)	

			#pragma unroll
			for (int q = 0; q < K; q++)
			{
				s[q] = index[q] != 0 ? state[abs(index[q])] : (TFloat)-1.0;	//gather (uncoalesced), abs(index) is a device-intrinsic
				s[q] = index[q] < 0 ? (TFloat)1.0 + s[q] : (TFloat)1.0 - s[q];
				km *= s[q];
			}

			//auxiliary rhs
			int p = cons.N + j + 1;
			am = state[p];			//load (coalesced)
			rhs[p] = am * km * km;	//store (coalesced)

			//rhs for variables - invoke templated loop
			KmiWrapper<TFloat, K, 0>::KmiFunc(rhs, index, s, km * cons.KNORM * am);
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
	void CudaSat1<TFloat>::SetDeviceConstants()
	{
		cons.KNORM = (TFloat)pow(2.0, -k);
		cons.N = n;
		cons.M = m;
		cons.STRIDE = stride;
		cons.GC = gC;
		cons.MEAN_AM = gMeanAm;
		cons.ALPHA = (TFloat)m / (TFloat)n;
	}


	template <typename TFloat>
	void CudaSat1<TFloat>::SetLaunchConfig()
	{
		int bs = 128;
		threads = dim3(bs);
		blocks = dim3((int)ceil((float)m / bs));
	}

	template <typename TFloat>
	CudaSat1<TFloat>::CudaSat1()
	{
		//gpu pointer init
		gC = 0;
		gBool = 0;
		gReduceBuf1 = 0;
		gReduceBuf2 = 0;
		gMeanAm = 0;

		n = m = k = 0;
		b = 0;
	}

	template<typename TFloat>
	void CudaSat1<TFloat>::SetProblem(const SatProblem& problem)
	{
		if (problem.Get_K() > 10 ) throw runtime_error("this solver only supports up to 10-SAT problems");

		n = problem.Get_N();
		m = problem.Get_M();
		k = problem.Get_K();

		Free();
		Allocate();
		SetDeviceConstants();
		SetLaunchConfig();
		InitProblem(problem);
	}


	template <typename TFloat>
	CudaSat1<TFloat>::~CudaSat1()
	{
		Free();	
	}

	template <typename TFloat>
	void CudaSat1<TFloat>::Allocate()
	{
		//calculate alloc size (align to 128 bytes)
		int block = 128 / sizeof(int);
		stride = (int)ceil((float)m / block) * block;

		CudaSafe(cudaMalloc(&gC, stride * k * sizeof(int)));
		CudaSafe(cudaMalloc(&gBool, sizeof(int) * m));
		CudaSafe(cudaMalloc(&gReduceBuf1, sizeof(int) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gReduceBuf2, sizeof(int) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gMeanAm, sizeof(TFloat)));

		//host alloc	
		C.resize(stride * k);
	}

	template <typename TFloat>
	void CudaSat1<TFloat>::Free()
	{
		SAFEDEL(gC);
		SAFEDEL(gBool);
		SAFEDEL(gReduceBuf1);
		SAFEDEL(gReduceBuf2);
		SAFEDEL(gMeanAm);
	}

	template <typename TFloat>
	inline int CudaSat1<TFloat>::GetC(int row, int col) { return C[row * stride + col]; }

	template <typename TFloat>
	inline void CudaSat1<TFloat>::SetC(int row, int col, int value) { C[row * stride + col] = value; }


	//initializes the SAT problem (clause optimizations)
	template <typename TFloat>
	void CudaSat1<TFloat>::InitProblem(const SatProblem& problem)
	{
		//reset clauses (creates zero padding)
		memset(C.data(), 0, C.size() * sizeof(int));

		//default ordering
		clauseOrder.resize(m);
		for (int j = 0; j < m; j++) clauseOrder[j] = j;

		//store: copy the clauses into the flat C matrix
		const int* storage = problem.GetLiterals();
		for (int j = 0; j < m; j++)
		{
			int jj = clauseOrder[j];
			int start = problem.GetClauseStart(jj);
			int length = problem.GetClauseLength(jj);
			for (int i = 0; i < length; i++) SetC(i, j, storage[start + i]);
		}

		//upload
		CudaSafe(cudaMemcpy(gC, C.data(), sizeof(int) * k * stride, cudaMemcpyHostToDevice));

	}

	template <typename TFloat>
	CudaSatState1<TFloat>* CudaSat1<TFloat>::MakeState() const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");
		return new CudaSatState1<TFloat>(n, m, clauseOrder);
	}

	template <typename TFloat>
	void CudaSat1<TFloat>::SetRandomState(IState& state, ISatRandom<TFloat>& random)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		//next n+m values: create uniform rand
		random.GenerateUniform(state + 1, n + m);

		//init values using the randoms (first n: *2-1, next m: +1)
		dim3 blocks2((int)ceil((float)(n + m + 1) / threads.x));
		KernelSetInitial1<TFloat> KERNEL_ARGS2(blocks2, threads)(n, m, state);

		CudaSafe(cudaGetLastError());
	}


	template <typename TFloat>
	int CudaSat1<TFloat>::Get_N() const { return n; }	//number of variables

	template <typename TFloat>
	int CudaSat1<TFloat>::Get_M() const { return m; }	//number of clauses

	template <typename TFloat>
	int CudaSat1<TFloat>::Get_K() const { return k; }	//max number of variables in a clause

	template<typename TFloat>
	TFloat CudaSat1<TFloat>::Get_B() const { return b; }	//max number of variables in a clause

	template<typename TFloat>
	void CudaSat1<TFloat>::Set_B(TFloat _b)  { b = _b; }	//max number of variables in a clause


	template <typename TFloat>
	int CudaSat1<TFloat>::GetClauseViolationCount(const IState& state) const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		//use state to evaluate clauses to boolean
		KernelCalculateClauses1<TFloat> KERNEL_ARGS2(blocks, threads)(cons, state, gBool, k);

		//reduce SUM on clause bools
		CudaReduceSum(m, gBool, gReduceBuf1, gReduceBuf2);

		//download results (single int value) -- blocking copy to force device sync
		int res;
		CudaSafe(cudaMemcpy(&res, gReduceBuf1, sizeof(int), cudaMemcpyDeviceToHost));
		return res;
	}

	template <typename TFloat>
	void CudaSat1<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		//reset the global mean am
		if (b > (TFloat)0) CudaSafe(cudaMemsetAsync(gMeanAm, 0, sizeof(TFloat)));

		dxdt.SetZero();
		cons.B = b * cons.KNORM * cons.KNORM; // premultiply the bias term with normalization

		//run the RHS calculation
		switch (k)
		{
		case 1: KernelCalculateRHSv1<TFloat, 1> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 2: KernelCalculateRHSv1<TFloat, 2> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 3: KernelCalculateRHSv1<TFloat, 3> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 4: KernelCalculateRHSv1<TFloat, 4> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 5: KernelCalculateRHSv1<TFloat, 5> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 6: KernelCalculateRHSv1<TFloat, 6> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 7: KernelCalculateRHSv1<TFloat, 7> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 8: KernelCalculateRHSv1<TFloat, 8> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 9: KernelCalculateRHSv1<TFloat, 9> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		case 10: KernelCalculateRHSv1<TFloat, 10> KERNEL_ARGS2(blocks, threads)(cons, dxdt, state); break;
		default: throw runtime_error("unsupported problem size");
		}

		if (b > zero)
		{
			//compute the additional sine term for each variable
			dim3 blocks2((int)ceil((float)(n + 1) / threads.x));
			KernelAdjustState1<TFloat> KERNEL_ARGS2(blocks2, threads)(cons, dxdt, state);
		}
	}

	//instantiate the class for float and double
	template class CudaSat1<double>;
	template class CudaSat1<float>;
}

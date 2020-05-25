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

#include "CudaSat2.h"
#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaReduce.h"
#include "../util/utils.h"

#include "CudaSatBase.h"

using namespace std;

namespace analogsat
{
	/*
	//simplified version, no templates on k or clauses_per_block -> params, and external shared
	template <typename TFloat>
	__global__ void KernelCalculateRHS2(TFloat* state, TFloat* rhs, int k, int clauses_per_block)
	{
		TFloat* sh_state = reinterpret_cast<TFloat*>(shmemx);
		const devconstSat<TFloat>& cons = GetConsSat<TFloat>();

		TFloat knorm = cons.KNORM;
		int* clauses = cons.GC;

		int i = blockIdx.x * blockDim.x + threadIdx.x;	//index in clause-literals
		int threadDivK = threadIdx.x / k;				//which clause (out of the clauses processed by this block)
		int threadModK = threadIdx.x - threadDivK * k;	//which literal within the clause

		//grab the desired section of clauses
		int idx = clauses[i]; //coalesced load

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
		TFloat km = knorm;
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
		if (threadModK == 0) sh_state[threadDivK] = km; //no two threads write the same address, no bank conflicts

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
		//result[i] = (TFloat)2.0 * kmi * km * am * sign; //coalesced
		TFloat re = (TFloat)2.0 * kmi * km * am * sign;
		if (idx >= 2) atomicAdd(&rhs[idx], re);
	}
	*/

	//simplified version, no templates on k or clauses_per_block -> params, and external shared
	template <typename TFloat>
	__global__ void KernelCalculateRHS2_mod(CudaSatArgs<TFloat> cons, TFloat* state, TFloat* rhs, int k, int clauses_per_block)
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
		TFloat re = (TFloat)2.0 * kmi * km * am * sign;

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
	void CudaSat2<TFloat>::SetDeviceConstants()
	{
		cons.KNORM = (TFloat)pow(2.0, -k);
		cons.N = n;
		cons.M = m_padded;
		cons.GC = gC;		
		cons.MEAN_AM = gMeanAm;
		cons.ALPHA = (TFloat)m / (TFloat)n;
	}

	template <typename TFloat>
	void CudaSat2<TFloat>::SetLaunchConfig()
	{
		threads = dim3(k * clauses_per_block);
		blocks = dim3((int)ceil((float)m_padded / clauses_per_block));
	}

	template <typename TFloat>
	CudaSat2<TFloat>::CudaSat2()
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

	template <typename TFloat>
	CudaSat2<TFloat>::~CudaSat2()
	{
		Free();
	}

	template<typename TFloat>
	void CudaSat2<TFloat>::SetProblem(const SatProblem& problem)
	{
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
	int CudaSat2<TFloat>::LCM(int a, int b)
	{
		int m = a;
		int n = b;

		while (m != n)
		{
			if (m < n) m = m + a;
			else n = n + b;
		}

		return m;
	}


	template <typename TFloat>
	void CudaSat2<TFloat>::Allocate()
	{
		//calculate blocksize = literals_per_block = least common multiple of warp size (32) and k
		int blocksize = LCM(32, k);
		if (blocksize > 1024) throw runtime_error("unsupported SAT size"); //CUDA limit: max 1024 threads per block

		clauses_per_block = blocksize / k;

		//total number of clauses with padding will be the nearest multiple of m to clauses_per_block	
		m_padded = (int)ceil((double)m / clauses_per_block) * clauses_per_block;

		CudaSafe(cudaMalloc(&gC, m_padded * k * sizeof(int)));
		CudaSafe(cudaMalloc(&gBool, sizeof(int) * m_padded));
		CudaSafe(cudaMalloc(&gReduceBuf1, sizeof(int) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gReduceBuf2, sizeof(int) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gMeanAm, sizeof(TFloat)));

		//convertBuf.resize(m); //here, so no need later	
		C.resize(m_padded * k);

	}

	template <typename TFloat>
	void CudaSat2<TFloat>::Free()
	{
		SAFEDEL(gC);
		SAFEDEL(gBool);
		SAFEDEL(gReduceBuf1);
		SAFEDEL(gReduceBuf2);
		SAFEDEL(gMeanAm);
	}

	template <typename TFloat>
	inline int CudaSat2<TFloat>::GetC(int row, int col) const { return C[col * k + row]; }

	template <typename TFloat>
	inline void CudaSat2<TFloat>::SetC(int row, int col, int value) { C[col * k + row] = value; }


	//initializes the SAT problem (clause optimizations, padding)
	template <typename TFloat>
	void CudaSat2<TFloat>::InitProblem(const SatProblem& problem)
	{
		//reset clauses (creates zero padding)
		memset(C.data(), 0, C.size() * sizeof(int));

		//order columns by first row index	
		//clauseOrder = sort_indices<vector<int>>(CC, &ColComparator); //this should be helpful for CPU by increasing cache locality

		//default ordering
		clauseOrder.resize(m);
		for (int j = 0; j < m; j++) clauseOrder[j] = j;

		////order columns (clauses) by length
		//clauseOrder.resize(m);
		//std::iota(clauseOrder.begin(), clauseOrder.end(), 0); //range
		//std::sort(clauseOrder.begin(), clauseOrder.end(), [&](size_t i1, size_t i2) {
		//	const auto& c1 = problem.GetClause((int)i1 + 1); //1-base index for problem clauses
		//	const auto& c2 = problem.GetClause((int)i2 + 1);
		//	return c1.Length() < c2.Length();
		//});


		//store: copy the clauses into the flat C matrix
		const int* storage = problem.GetLiterals();
		for (int j = 0; j < m; j++)
		{
			int jj = clauseOrder[j];
			int start = problem.GetClauseStart(jj);
			int length = problem.GetClauseLength(jj);
			for (int i = 0; i < length; i++)
			{
				int idx = storage[start + i];
				if (idx < 0) idx--; else idx++; //+1 for shifting variable index up (idx==1 is the TRUE constant) | the rest are zero, that does not shift
				SetC(i, j, idx);
			}
		}

		for (int j = m; j < m_padded; j++) //clause padding with all-TRUE clauses
		{
			for (int i = 0; i < k; i++) SetC(i, j, 1);
		}

		//upload
		CudaSafe(cudaMemcpy(gC, C.data(), sizeof(int) * k * m_padded, cudaMemcpyHostToDevice));
	}

	template <typename TFloat>
	CudaSatState2<TFloat>* CudaSat2<TFloat>::MakeState() const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");
		return new CudaSatState2<TFloat>(n, m, clauseOrder, m_padded - m); //pass without clause ordering, making it an internal state only
	}


	template <typename TFloat>
	void CudaSat2<TFloat>::SetRandomState(IState& state, ISatRandom<TFloat>& random)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		cudaMemcpy(state, &minusone, sizeof(double), cudaMemcpyHostToDevice);	//always false
		cudaMemcpy(state + 1, &one, sizeof(double), cudaMemcpyHostToDevice);	//always true
		
		//uniform rand		
		random.GenerateUniform(state + 2, n + m);
		CudaSafe(cudaGetLastError());

		//transform randoms
		dim3 blocks2((int)ceil((float)(n + m + 2) / threads.x));
		KernelSetInitial2<TFloat> KERNEL_ARGS2(blocks2, threads)(n, m, state);

		CudaSafe(cudaGetLastError());
	}

	template <typename TFloat>
	int CudaSat2<TFloat>::Get_N() const { return n; }	//number of variables

	template <typename TFloat>
	int CudaSat2<TFloat>::Get_M() const { return m; }	//number of clauses

	template <typename TFloat>
	int CudaSat2<TFloat>::Get_K() const { return k; }	//max number of variables in a clause

	template<typename TFloat>
	TFloat CudaSat2<TFloat>::Get_B() const { return b; }	//max number of variables in a clause

	template<typename TFloat>
	void CudaSat2<TFloat>::Set_B(TFloat _b)  { b = _b; }	//max number of variables in a clause

	template <typename TFloat>
	int CudaSat2<TFloat>::GetClauseViolationCount(const IState& state) const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		int shmem = k * clauses_per_block * sizeof(int);
		KernelCalculateClauses2<TFloat> KERNEL_ARGS3(blocks, threads, shmem)(cons, state, gBool, k, clauses_per_block);

		//reduce SUM on clause bools
		CudaReduceSum(m_padded, gBool, gReduceBuf1, gReduceBuf2);

		//download results
		int res;
		CudaSafe(cudaMemcpy(&res, gReduceBuf1, sizeof(int), cudaMemcpyDeviceToHost));
		return res;
	}

	template <typename TFloat>
	void CudaSat2<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");
			
		//reset the global mean am
		if (b > (TFloat)0) CudaSafe(cudaMemsetAsync(gMeanAm, 0, sizeof(TFloat)));
		
		dxdt.SetZero();
		cons.B = b * cons.KNORM * cons.KNORM; // premultiply the bias term with normalization		

		//call the RHS kernel
		int shmem2 = k * clauses_per_block * sizeof(TFloat);
		KernelCalculateRHS2_mod<TFloat> KERNEL_ARGS3(blocks, threads, shmem2)(cons, state, dxdt, k, clauses_per_block);
		
		if (b > (TFloat)0)
		{
			//add the sine term for each variable
			dim3 blocks2((int)ceil((float)(n + 2) / threads.x));
			KernelAdjustState2<TFloat> KERNEL_ARGS2(blocks2, threads)(cons, dxdt, state);
		}
	}

	//instantiate the class for float and double
	template class CudaSat2<double>;
	template class CudaSat2<float>;
}

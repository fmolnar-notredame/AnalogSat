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

#include "CudaSat3.h"
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
	__global__ void KernelCalculateRHS3(CudaSatArgs<TFloat> cons, TFloat* state, TFloat* rhs, TFloat* collect, int k, int clauses_per_block)
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
		if (idx >= 2) collect[i] = (TFloat)2.0 * kmi * km * am * sign; //coalesced
	}


	template <typename TFloat>
	void CudaSat3<TFloat>::SetDeviceConstants()
	{		
		cons.KNORM = (TFloat)pow(2.0, -k);
		cons.N = n;
		cons.M = m_padded;
		cons.GC = gC;
		cons.ALPHA = (TFloat)m / (TFloat)n;
	}

	template <typename TFloat>
	void CudaSat3<TFloat>::SetLaunchConfig()
	{
		threads = dim3(k * clauses_per_block);
		blocks = dim3((int)ceil((float)m_padded / clauses_per_block));

		threads_collect = dim3(64);
		int vars_per_block = threads_collect.x / 32; //one warp per variable
		blocks_collect = dim3((int)ceil((double)(n + 2) / vars_per_block)*vars_per_block);
	}

	template <typename TFloat>
	CudaSat3<TFloat>::CudaSat3()
	{
		//gpu pointer init
		gC = 0;
		gBool = 0;
		gReduceBuf1 = 0;
		gReduceBuf2 = 0;
		gStartVar = 0;
		gEndVar = 0;
		gCollect = 0;
		gCn = 0;
		gAux1 = 0;
		gAux2 = 0;

		n = m = k = 0;
		b = 0;
	}

	template <typename TFloat>
	CudaSat3<TFloat>::~CudaSat3()
	{
		Free();		
	}

	template<typename TFloat>
	void CudaSat3<TFloat>::SetProblem(const SatProblem& problem)
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
	int CudaSat3<TFloat>::LCM(int a, int b)
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
	void CudaSat3<TFloat>::Allocate()
	{
		//calculate blocksize = literals_per_block = least common multiple of warp size (32) and k
		int blocksize = LCM(32, k);
		if (blocksize > 1024) throw runtime_error("unsupported SAT size"); //CUDA limit: max 1024 threads per block

		clauses_per_block = blocksize / k;

		//total number of clauses with padding will be the nearest multiple of m to clauses_per_block
		m_padded = (int)ceil((double)m / clauses_per_block) * clauses_per_block;

		//Note, the total state vector size will be n + 2 + m_padded
		//padding for aux variables: only used by padded clauses. Those should always be satisfied, so RHS=0 on them

		CudaSafe(cudaMalloc(&gC, m_padded * k * sizeof(int)));
		CudaSafe(cudaMalloc(&gBool, sizeof(int) * m_padded));
		CudaSafe(cudaMalloc(&gReduceBuf1, sizeof(int) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gReduceBuf2, sizeof(int) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gAux1, sizeof(TFloat) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gAux2, sizeof(TFloat) * MAX_BLOCK_DIM_SIZE));

		CudaSafe(cudaMalloc(&gStartVar, sizeof(int)* (n + 2)));
		CudaSafe(cudaMalloc(&gEndVar, sizeof(int)* (n + 2)));
		CudaSafe(cudaMalloc(&gCollect, m_padded * k * sizeof(TFloat)));

		//convertBuf.resize(m); //here, so no need later	
		C.resize(m_padded * k);
	}

	template <typename TFloat>
	void CudaSat3<TFloat>::Free()
	{
		SAFEDEL(gC);
		SAFEDEL(gBool);
		SAFEDEL(gReduceBuf1);
		SAFEDEL(gReduceBuf2);
		SAFEDEL(gAux1);
		SAFEDEL(gAux2);

		SAFEDEL(gStartVar);
		SAFEDEL(gEndVar);
		SAFEDEL(gCollect);
		SAFEDEL(gCn);
	}

	template <typename TFloat>
	inline int CudaSat3<TFloat>::GetC(int row, int col) const { return C[col * k + row]; }

	template <typename TFloat>
	inline void CudaSat3<TFloat>::SetC(int row, int col, int value) { C[col * k + row] = value; }


	//initializes the SAT problem (clause optimizations, padding)
	template <typename TFloat>
	void CudaSat3<TFloat>::InitProblem(const SatProblem& problem)
	{
		//reset clauses (creates zero padding)
		memset(C.data(), 0, C.size() * sizeof(int));

		//order columns by first row index	
		//clauseOrder = sort_indices<vector<int>>(CC, &ColComparator); //this should be helpful for CPU by increasing cache locality

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
			for (int i = 0; i < length; i++)
			{
				int idx = storage[start + i];
				if (idx < 0) idx--; else idx++; //+1 for shifting variable index up (idx==1 is the TRUE constant) | the rest are zero, that does not shift
				SetC(i, j, idx);
			}
		}

		for (int j = m; j < m_padded; j++) //padding with all-TRUE clauses
		{
			for (int i = 0; i < k; i++) SetC(i, j, 1);
		}

		//upload
		CudaSafe(cudaMemcpy(gC, C.data(), sizeof(int) * k * m_padded, cudaMemcpyHostToDevice));

		// deterministic gather ----------------------------------------------------------
		//first calculate for each variable, which clause-literal index they participate in.
		vector<vector<int>> Cn(n + 2);

		for (int j = 0; j < m_padded; j++)
		{
			for (int i = 0; i < k; i++)
			{
				int item = GetC(i, j);
				int idx = item < 0 ? -item : item;
				Cn[idx].push_back(j * k + i); //index of the literal
			}
		}


		//build hCn --> gCn, and the bounds
		vector<int> startVar(n + 2);
		vector<int> endVar(n + 2);
		vector<int> hCn;
		int loc = 0;
		for (int i = 0; i < n + 2; i++)
		{
			int len = (int)Cn[i].size();
			int padded = (int)ceil((double)len / 32.0) * 32;

			//store
			for (int j = 0; j < len; j++) hCn.push_back(Cn[i][j]);
			for (int j = len; j < padded; j++) hCn.push_back(-1);

			startVar[i] = loc;
			endVar[i] = loc + padded;
			loc += padded;
		}

		//re-allocate gCn (its size is data-dependent)
		SAFEDEL(gCn);
		CudaSafe(cudaMalloc(&gCn, sizeof(int) * loc));

		//upload
		CudaSafe(cudaMemcpy(gStartVar, startVar.data(), (n + 2) * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafe(cudaMemcpy(gEndVar, endVar.data(), (n + 2) * sizeof(int), cudaMemcpyHostToDevice));
		CudaSafe(cudaMemcpy(gCn, hCn.data(), loc * sizeof(int), cudaMemcpyHostToDevice));
	}

	template <typename TFloat>
	CudaSatState2<TFloat>* CudaSat3<TFloat>::MakeState() const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");
		return new CudaSatState2<TFloat>(n, m, clauseOrder, m_padded - m);
	}

	template <typename TFloat>
	void CudaSat3<TFloat>::SetRandomState(IState& state, ISatRandom<TFloat>& random)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		cudaMemcpy(state, &minusone, sizeof(double), cudaMemcpyHostToDevice);	//always false
		cudaMemcpy(state + 1, &one, sizeof(double), cudaMemcpyHostToDevice);		//always true

		//uniform rand
		random.GenerateUniform(state + 2, n + m);

		//transform randoms
		dim3 blocks2((int)ceil((float)(n + m + 2) / threads.x));
		KernelSetInitial2<TFloat> KERNEL_ARGS2(blocks2, threads)(n, m, state);

		CudaSafe(cudaGetLastError());
	}



	template <typename TFloat>
	int CudaSat3<TFloat>::Get_N() const { return n; }	//number of variables

	template <typename TFloat>
	int CudaSat3<TFloat>::Get_M() const { return m; }	//number of clauses

	template <typename TFloat>
	int CudaSat3<TFloat>::Get_K() const { return k; }	//max number of variables in a clause

	template<typename TFloat>
	TFloat CudaSat3<TFloat>::Get_B() const { return b; }	//max number of variables in a clause

	template<typename TFloat>
	void CudaSat3<TFloat>::Set_B(TFloat _b)  { b = _b; }	//max number of variables in a clause

	template <typename TFloat>
	int CudaSat3<TFloat>::GetClauseViolationCount(const IState& state) const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		////DEBUG
		//vector<TFloat> temp(state.GetAllocSize());
		//cudaMemcpy(temp.data(), state, sizeof(TFloat)*state.GetAllocSize(), cudaMemcpyDeviceToHost);
		//vector<int> v(m_padded);
		//int viola = 0;
		//for (int j = 0; j < m_padded; j++)
		//{
		//	for (int i = 0; i < k; i++)
		//	{
		//		int lit = GetC(i, j);
		//	}

		//	bool clause = false;
		//	for (int i = 0; i < k; i++)
		//	{
		//		int item = GetC(i, j);				
		//		int index = item < 0 ? -item : item;

		//		bool lit = temp[index] > 0;
		//		if (item < 0) lit = !lit;
		//		clause |= lit;
		//		if (clause) break;
		//	}
		//	v[j] = clause ? 0 : 1;
		//	viola += v[j];
		//}
		////----------------


		int shmem = k * clauses_per_block * sizeof(int);
		KernelCalculateClauses2<TFloat> KERNEL_ARGS3(blocks, threads, shmem)(cons, state, gBool, k, clauses_per_block);

		//vector<int> bb(m_padded);
		//cudaMemcpy(bb.data(), gBool, sizeof(int)*m_padded, cudaMemcpyDeviceToHost);
		//for (int j = 0; j < m_padded; j++)
		//{
		//	if (bb[j] != v[j]) printf("Clause mismatch: %d\n", j);
		//}

		//reduce SUM on clause bools
		CudaReduceSum<int>(m_padded, gBool, gReduceBuf1, gReduceBuf2);

		//download results
		int res;
		CudaSafe(cudaMemcpy(&res, gReduceBuf1, sizeof(int), cudaMemcpyDeviceToHost));
		return res;
	}

	template <typename TFloat>
	void CudaSat3<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		//step 0: reduce sum auxiliary variables (deterministic sum)
		if (b > (TFloat)0) CudaReduceSum<TFloat>(m, state + n + 2, gAux1, gAux2);

		dxdt.SetZero();
		cons.B = b * cons.KNORM * cons.KNORM; // premultiply the bias term with normalization
		cons.MEAN_AM = gAux1;

		//step 1: calculate RHS terms
		int shmem = k * clauses_per_block * sizeof(TFloat);
		KernelCalculateRHS3<TFloat> KERNEL_ARGS3(blocks, threads, shmem)(cons, state, dxdt, gCollect, k, clauses_per_block);

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
	template class CudaSat3<double>;
	template class CudaSat3<float>;
}

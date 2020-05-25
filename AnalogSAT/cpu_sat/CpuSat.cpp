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


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "CpuSat.h"
#include "CpuSatStateImpl.h"
#include "cpu_helper.h"
#include "../util/utils.h"

using namespace std;

namespace analogsat
{
	//initialize the SAT ODE from a given problem
	template<typename TFloat>
	CpuSat<TFloat>::CpuSat()
	{
		n = m = k = 0;
		b = 0; //turn off b values by default		
	}


	template<typename TFloat>
	CpuSat<TFloat>::~CpuSat()
	{
	}

	template<typename TFloat>
	void CpuSat<TFloat>::SetProblem(const SatProblem& problem)
	{
		n = problem.Get_N();
		m = problem.Get_M();
		k = problem.Get_K();

		alpha = (TFloat)m / (TFloat)n;

		kmi.resize(k);
		si.resize(k);
		C.resize(m * k);

		InitProblem(problem);
	}

	template<typename TFloat>
	void CpuSat<TFloat>::GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		const vector<TFloat>& x = state.GetState();
		vector<TFloat>& rhs = dxdt.GetState();

		//zero out the RHS vector
		dxdt.SetZero();

		//precompute the scaled b prefactor
		bb = b * knorm * knorm;

		switch (k)
		{
		case 2: CalculateAllEx<2>(x, rhs); break;
		case 3: CalculateAllEx<3>(x, rhs); break;
		case 4: CalculateAllEx<4>(x, rhs); break;
		case 5: CalculateAllEx<5>(x, rhs); break;
		case 6: CalculateAllEx<6>(x, rhs); break;
		case 7: CalculateAllEx<7>(x, rhs); break;
		case 8: CalculateAllEx<8>(x, rhs); break;
		case 9: CalculateAllEx<9>(x, rhs); break;
		case 10: CalculateAllEx<10>(x, rhs); break;
		default: CalculateAll(x, rhs); break;
		}

		//ensure: the constant FALSE variable does not change
		rhs[0] = zero;
	}


	template<typename TFloat>
	int CpuSat<TFloat>::Get_N() const { return n; }	//number of variables

	template<typename TFloat>
	int CpuSat<TFloat>::Get_M()	const { return m; }	//number of clauses

	template<typename TFloat>
	int CpuSat<TFloat>::Get_K() const { return k; }	//max number of variables in a clause

	template<typename TFloat>
	TFloat CpuSat<TFloat>::Get_B() const { return b; }	//max number of variables in a clause

	template<typename TFloat>
	void CpuSat<TFloat>::Set_B(TFloat _b)  { b = _b; }	//max number of variables in a clause


	template<typename TFloat>
	CpuSatStateImpl<TFloat>* CpuSat<TFloat>::MakeState() const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");
		return new CpuSatStateImpl<TFloat>(n, m, clauseOrder);
	}

	//creates a suitable random initial state, ensuring correct vector size
	//usage is optional, user may provide their own initial state	
	template<typename TFloat>
	void CpuSat<TFloat>::SetRandomState(IState& state, ISatRandom<TFloat>& random)
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		vector<TFloat>& stat = state.GetState();
		stat.resize(n + m + 1); //n variables, m auxiliary variables, 1 constant
		stat[0] = minusone;		//constant false
		random.GenerateUniform(stat.data() + 1, n + m);
		for (int i = 1; i < n + 1; i++) stat[i] = stat[i] * two - one;
		for (int i = n + 1; i < n + m + 1; i++) stat[i] = stat[i] + one;
	}

	template<typename TFloat>
	void CpuSat<TFloat>::CalculateClauses(const std::vector<TFloat>& state) const
	{
		if (k == 0) throw runtime_error("SAT problem has not been set");

		vector<bool>& v = const_cast<vector<bool>&>(vars); // clause calculation is const to the public side
		v.resize(m);

		//evaluate clauses
		bool clause;
		int item;
		int index;
		bool lit;

		for (int j = 0; j < m; j++)
		{
			clause = false;
			for (int i = 0; i < k; i++)
			{
				item = GetC(i, j);
				index = item < 0 ? -item : item;

				lit = state[index] > zero;
				if (item < 0) lit = !lit;
				clause |= lit;
				if (clause) break;
			}
			v[j] = clause ? 0 : 1;
		}
	}

	template<typename TFloat>
	int CpuSat<TFloat>::GetClauseViolationCount(const IState& state) const
	{
		CalculateClauses(state.GetState());

		int count = 0;
		for (int j = 0; j < m; j++) count += vars[j];
		return count;
	}


	//extract the index and the sign from a given clause-index, and get the state
	template<typename TFloat>
	void CpuSat<TFloat>::GetStateIndexSign(const vector<TFloat>& state, TFloat& s, int& index, TFloat& sign)
	{
		if (index < 0) { index = -index; sign = minusone; s = state[index]; }
		else { sign = one; s = state[index]; }
	}

	template<typename TFloat>
	void CpuSat<TFloat>::GetIndexSign(int& index, TFloat& sign)
	{
		if (index < 0) { index = -index; sign = minusone; }
		else { sign = one; }
	}



#define PI 3.141592653589793
#define PI_OVER_2 1.570796326794897

	template <typename TFloat>
	template <int K>
	void CpuSat<TFloat>::CalculateAllEx(const vector<TFloat>& state, vector<TFloat>& rhs)
	{
		int index[K];
		TFloat s[K];
		TFloat km, am = 0;
		TFloat sum = 0;

		for (int j = 0; j < m; j++)
		{
			km = knorm;

			for (int q = 0; q < K; q++) index[q] = GetC(q, j);  //cache-aligned

			for (int q = 0; q < K; q++)	//local compute
			{
				s[q] = index[q] != 0 ? state[abs(index[q])] : (TFloat)-1.0;		//gather
				s[q] = index[q] < 0 ? (TFloat)1.0 + s[q] : (TFloat)1.0 - s[q];
				km *= s[q];
			}

			//take care of the auxiliary variable
			int p = n + 1 + j;
			am = state[p];
			rhs[p] = am * km * km;

			//reduce am
			sum += am;

			//rhs for variables - invoke templated loop
			CpuKmiWrapper<TFloat, K, 0>::KmiFunc(rhs.data(), index, s, km * knorm * am);
		}

		// second term
		if (bb > zero)
		{
			TFloat a = sum / m;
			for (int i = 1; i < n + 1; i++)
			{
				rhs[i] += a * alpha * bb * (TFloat)PI_OVER_2 * sin(state[i] * (TFloat)PI);
			}
		}

	}

	//calculates rhs - special case for k=3, unrolled loops and state cache - KEEP THIS FOR ILLUSTRATION PURPOSES ONLY
	template<typename TFloat>
	void CpuSat<TFloat>::CalculateAll3(const vector<TFloat>& state, vector<TFloat>& dxdt)
	{
		int index1, index2, index3;
		TFloat sign1, sign2, sign3;
		TFloat km, am;
		TFloat s1, s2, s3;

		//mult for Km
		for (int j = 0; j < m; j++)
		{
			km = knorm;

			//load (cache-aligned)
			index1 = GetC(0, j);
			index2 = GetC(1, j);
			index3 = GetC(2, j);

			//gather + multiply under latency
			GetStateIndexSign(state, s1, index1, sign1);
			s1 = one - s1 * sign1;
			km *= s1;

			GetStateIndexSign(state, s2, index2, sign2);
			s2 = one - s2 * sign2;
			km *= s2;

			GetStateIndexSign(state, s3, index3, sign3);
			s3 = one - s3 * sign3;
			km *= s3;

			//auxiliary rhs
			int p = n + 1 + j;
			am = state[p];
			dxdt[p] = am * km;

			//rhs for variables		
			dxdt[index1] += two * km * knorm * s2 * s3 * am * sign1;

			dxdt[index2] += two * km * knorm * s1 * s3 * am * sign2;

			dxdt[index3] += two * km * knorm * s1 * s2 * am * sign3;
		}
	}

	//compute rhs, generic case (for K>10)
	template<typename TFloat>
	void CpuSat<TFloat>::CalculateAll(const vector<TFloat>& state, vector<TFloat>& dxdt)
	{
		int index;
		TFloat sign;
		TFloat km, am;
		TFloat temp;
		TFloat sum = 0;

		//mult for Km
		for (int j = 0; j < m; j++)
		{
			km = knorm;

			for (int i = 0; i < k; i++)
			{
				index = GetC(i, j); //cache-aligned
				GetStateIndexSign(state, si[i], index, sign);

				//gather + multiply under latency
				temp = one - si[i] * sign;
				km *= temp;
				kmi[i] = temp;
			}

			//take care of the auxiliary variable
			int p = n + 1 + j;
			am = state[p];
			dxdt[p] = am * km * km;

			//reduce am
			sum += am;

			//the current km will participate in the sums of those i variables who we just looped over
			//loop over again to add these to their sums			
			for (int i = 0; i < k; i++)
			{
				index = GetC(i, j); //cache-aligned (maybe still in cache from last loop)				
				GetIndexSign(index, sign);

				//prepare the kmi*km product
				temp = km;
				for (int ii = 0; ii < k; ii++)
					temp *= (ii == i) ? one : kmi[ii]; //branch-free multiply, kmi[] should be in L1 cache

				//gather-scatter
				dxdt[index] += two * temp * knorm * am * sign;
			}
		}

		// second term
		if (bb > zero)
		{
			TFloat a = sum / m;
			for (int i = 1; i < n + 1; i++)
			{
				dxdt[i] += a * alpha * bb * (TFloat)PI_OVER_2 * sin(state[i] * (TFloat)PI);
			}
		}
	}

	//initializes the SAT problem (clause optimizations)
	template<typename TFloat>
	void CpuSat<TFloat>::InitProblem(const SatProblem& problem)
	{
		knorm = TFloat(pow(2.0, -k)); //no bit-shift trickery, because what if k>32...		

		//reset clauses (creates zero padding)
		memset(C.data(), 0, C.size() * sizeof(int));

		//order columns by first row index	
		//clauseOrder = sort_indices<vector<int>>(CC, &ColComparator); //this should be helpful for CPU by increasing cache locality | Re: not really

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

	}

	//row-major order
	//inline int GetC(int row, int col) { return C[row * m + col]; }
	//inline void SetC(int row, int col, int value) { C[row * m + col] = value; }

	//col-major order
	template<typename TFloat>
	inline int CpuSat<TFloat>::GetC(int row, int col) const { return C[col * k + row]; }

	template<typename TFloat>
	inline void CpuSat<TFloat>::SetC(int row, int col, int value) { C[col * k + row] = value; }

	//instantiation
	template class CpuSat<float>;
	template class CpuSat<double>;
	template class CpuSat<long double>;
}

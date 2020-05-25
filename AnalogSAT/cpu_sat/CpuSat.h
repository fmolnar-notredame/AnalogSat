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


#ifndef ANALOGSAT_CPUSAT_ORIG_H
#define ANALOGSAT_CPUSAT_ORIG_H

#include <vector>
#include <random>
#include <thread>

#include "../problem/SatProblem.h"
#include "../solver/ISat.h"
#include "CpuSatStateImpl.h"

namespace analogsat
{
	// The ODE representation of the SAT on the host/CPU
	// implements the RHS of the equation
	// supports incomplete clauses
	template<typename TFloat>
	class CpuSat : public ISat<TFloat, CpuODEState<TFloat>>
	{
	private:

		int k, n, m;			//size of clauses, number of variables, number of clauses
		std::vector<int> C;		//clause matrix, rows of column-compressed data, with sign, and zeros for incomplete clauses
		std::vector<int> clauseOrder;	//ordering of the original clauses

		//variables for RHS calculation
		TFloat knorm;				//normalization factor
		std::vector<TFloat> kmi;	//kmi product
		std::vector<TFloat> si;		//spin variable

		std::vector<bool> vars;			//temp buffer for computing clause violations

		TFloat b, bb;	//bias term, original and scaled values
		TFloat alpha;	//clause to variable ratio

		//explicit-type constants
		const TFloat zero = (TFloat)0.0;
		const TFloat one = (TFloat)1.0;
		const TFloat two = (TFloat)2.0;
		const TFloat minusone = (TFloat)-1.0;

	public:

		typedef CpuSatStateImpl<TFloat> State;
		typedef ISatState<TFloat, CpuODEState<TFloat>> IState;
		typedef IODEState<TFloat, CpuODEState<TFloat>> IBasicState;

		CpuSat();
		~CpuSat();

		//implement IODE and ISat interfaces --------------

		//initialize the SAT ODE from a given problem
		void SetProblem(const SatProblem& problem) override;

		//RHS
		void GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time) override;

		int Get_N() const override;	//number of variables
		int Get_M()	const override;	//number of clauses
		int Get_K() const override;	//max number of variables in a clause

		//get the sine term prefactor
		TFloat Get_B() const override;

		//set the sine term prefactor
		void Set_B(TFloat _b) override;

		//create a compatible ODE state vector
		State* MakeState() const override;

		//sets a random state suitable for initial condition
		void SetRandomState(IState& state, ISatRandom<TFloat>& random) override;

		//count the clause violations
		int GetClauseViolationCount(const IState& state) const override;

		using ISat<TFloat, CpuODEState<TFloat>>::GetClauseViolationCount;

	private:

		//extract the index and the sign from a given clause-index, and get the state
		inline void GetStateIndexSign(const std::vector<TFloat>& state, TFloat& s, int& index, TFloat& sign);
		inline void GetIndexSign(int& index, TFloat& sign);

		//evaluate the state into clause violations
		void CalculateClauses(const std::vector<TFloat>& state) const;

		//calculates rhs - special case for k=3, manually unrolled loops and state cache
		void CalculateAll3(const std::vector<TFloat>& state, std::vector<TFloat>& dxdt);

		//compute rhs, generic case. 	
		void CalculateAll(const std::vector<TFloat>& state, std::vector<TFloat>& dxdt);

		template <int K>
		void CalculateAllEx(const std::vector<TFloat>& state, std::vector<TFloat>& rhs);

		//initializes the SAT problem (clause optimizations)
		void InitProblem(const SatProblem& problem);

		inline int GetC(int row, int col) const;
		inline void SetC(int row, int col, int value);

	};
}

#endif
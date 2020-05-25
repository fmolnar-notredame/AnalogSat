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


#ifndef ANALOGSAT_CUDA_SAT2_H
#define ANALOGSAT_CUDA_SAT2_H

#include <vector>

#include "../solver/ISat.h"
#include "../problem/SatProblem.h"
#include "../cuda_base/CudaRandom.h"
#include "../cuda_base/CudaSatState2.h"
#include "../cuda_base/CudaSatStructs.h"

namespace analogsat
{

	//version 2 of the SAT implementation
	//- k-SAT supported up to k <= 256
	//- solution is non-deterministic w.r.t. the same input data
	//- it is quite fast
	template <typename TFloat>
	class CudaSat2 : public ISat<TFloat, CudaODEState<TFloat>>
	{
	protected:

		int n, m, k;					//number of variables, number of clauses, max number of literals in a clause
		std::vector<int> C;				//clause matrix, columns one after the other (same on GPU!)
		std::vector<bool> vars;			//use these variables in verification
		std::vector<int> clauseOrder;	//ordering of the original clauses		

		//constants
		const TFloat zero = (TFloat)0.0;
		const TFloat one = (TFloat)1.0;
		const TFloat two = (TFloat)2.0;
		const TFloat minusone = (TFloat)-1.0;

		//gpu arrays
		int* gC = 0;			//clauses		
		int* gBool = 0;			//holds true/false for each clause
		int* gReduceBuf1 = 0;	//holds the temporary reduction results
		int* gReduceBuf2 = 0;	//holds the temporary reduction results
		TFloat* gMeanAm;		//mean am value (atomic global, single value)

		//gpu interfacing
		virtual void SetDeviceConstants();
		virtual void SetLaunchConfig();
		virtual void Allocate();
		virtual void Free();

		//initializes the SAT problem (clause optimizations) and upload to gpu
		virtual void InitProblem(const SatProblem& problem);

		//clause access on host
		inline int GetC(int row, int col) const;
		inline void SetC(int row, int col, int value);

		int LCM(int a, int b); //least common multiple of two numbers

		//locals
		CudaSatArgs<TFloat> cons;
		dim3 blocks, threads;
		int blocksize, clauses_per_block;
		int m_padded;		
		TFloat b;		

	public:

		//state types for this SAT implementation
		typedef CudaSatState2<TFloat> State;
		typedef ISatState<TFloat, CudaODEState<TFloat>> IState;
		typedef IODEState<TFloat, CudaODEState<TFloat>> IBasicState;

		CudaSat2();
		~CudaSat2() override;

		//initialize the SAT ODE from a given problem
		virtual void SetProblem(const SatProblem& problem) override;

		//allocate a new state vector
		virtual State* MakeState() const override;		

		//creates a suitable random initial state
		virtual void SetRandomState(IState& state, ISatRandom<TFloat>& random) override;

		//calculate RHS by kernel invocation. device pointers expected, dxdt zeroed out expected
		virtual void GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time) override;

		virtual int Get_N() const override; //number of variables
		virtual int Get_M() const override; //number of clauses
		virtual int Get_K() const override; //max number of variables in a clause
		
		//get the sine term prefactor
		virtual TFloat Get_B() const override;

		//set the sine term prefactor
		virtual void Set_B(TFloat _b) override;

		//calculate the number of violated clauses
		virtual int GetClauseViolationCount(const IState& state) const override;

		using ISat<TFloat, CudaODEState<TFloat>>::GetClauseViolationCount;

	};

}

#endif

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


#ifndef ANALOGSAT_CUDA_SAT3_TANH_H
#define ANALOGSAT_CUDA_SAT3_TANH_H

#include <vector>

#include "../solver/ISat.h"
#include "../problem/SatProblem.h"
#include "../cuda_base/CudaSatState2.h"

#include "CudaSat3.h"

namespace analogsat
{
	//Tanh CTDS, v3
	//deterministic RHS calculation, slower	
	template <typename TFloat>
	class CudaSatTanh3 : public CudaSat3<TFloat>
	{
		using CudaSat3<TFloat>::b;
		using CudaSat3<TFloat>::k;
		using CudaSat3<TFloat>::n;
		using CudaSat3<TFloat>::m;		
		using CudaSat3<TFloat>::m_padded;
		using CudaSat3<TFloat>::gC;
		using CudaSat3<TFloat>::gCn;
		using CudaSat3<TFloat>::cons;
		using CudaSat3<TFloat>::one;
		using CudaSat3<TFloat>::blocks;
		using CudaSat3<TFloat>::threads;
		using CudaSat3<TFloat>::blocks_collect;
		using CudaSat3<TFloat>::threads_collect;
		using CudaSat3<TFloat>::clauses_per_block;
		using CudaSat3<TFloat>::Free;		
		using CudaSat3<TFloat>::gAux1;
		using CudaSat3<TFloat>::gAux2;
		using CudaSat3<TFloat>::gStartVar;
		using CudaSat3<TFloat>::gEndVar;
		using CudaSat3<TFloat>::gCollect;

	protected:

		TFloat q;

	public:

		//state types for this SAT implementation
		typedef CudaSatState2<TFloat> State;
		typedef ISatState<TFloat, CudaODEState<TFloat>> IState;
		typedef IODEState<TFloat, CudaODEState<TFloat>> IBasicState;

		//initialize the SAT ODE from a given problem
		CudaSatTanh3();
		~CudaSatTanh3() override;

		//calculate RHS by kernel invocation
		void GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time) override;

		//get the q parameter of the Tanh
		TFloat Get_Q() const;

		//set the q parameter for the Tanh
		void Set_Q(TFloat _q);
	};

}

#endif

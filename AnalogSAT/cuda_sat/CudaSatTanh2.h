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


#ifndef ANALOGSAT_CUDA_SAT2_TANH_H
#define ANALOGSAT_CUDA_SAT2_TANH_H

#include <vector>
#include <algorithm>

#include "../solver/ISat.h"
#include "../problem/SatProblem.h"
#include "../cuda_base/CudaSatState2.h"

#include "CudaSat2.h"

namespace analogsat
{

	template <typename TFloat>
	class CudaSatTanh2 : public CudaSat2<TFloat>
	{
		using CudaSat2<TFloat>::b;
		using CudaSat2<TFloat>::k;
		using CudaSat2<TFloat>::n;
		using CudaSat2<TFloat>::m;
		using CudaSat2<TFloat>::cons;
		using CudaSat2<TFloat>::gMeanAm;
		using CudaSat2<TFloat>::m_padded;
		using CudaSat2<TFloat>::gC;
		using CudaSat2<TFloat>::one;
		using CudaSat2<TFloat>::blocks;
		using CudaSat2<TFloat>::threads;
		using CudaSat2<TFloat>::clauses_per_block;
		using CudaSat2<TFloat>::Free;		

	protected:

		TFloat q;

	public:

		//state types for this SAT implementation
		typedef CudaSatState2<TFloat> State;
		typedef ISatState<TFloat, CudaODEState<TFloat>> IState;
		typedef IODEState<TFloat, CudaODEState<TFloat>> IBasicState;

		CudaSatTanh2();
		~CudaSatTanh2() override;

		//calculate RHS by kernel invocation. device pointers expected, dxdt zeroed out expected
		virtual void GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time) override;

		//get the q parameter of the Tanh
		TFloat Get_Q() const;

		//set the q parameter for the Tanh
		void Set_Q(TFloat _q);
	};
}

#endif

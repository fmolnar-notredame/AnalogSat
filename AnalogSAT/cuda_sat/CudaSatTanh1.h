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


#ifndef ANALOGSAT_CUDA_SAT1_TANH_H
#define ANALOGSAT_CUDA_SAT1_TANH_H

#include <vector>
#include "curand.h"

#include "../solver/ISat.h"
#include "../problem/SatProblem.h"
#include "../cuda_base/CudaSatState1.h"
#include "../cuda_base/CudaRandom.h"

#include "CudaSat1.h"

namespace analogsat
{
	//version 1 of the SAT implementation, Tanh CTDS
	//- k-SAT supported up to k <= 10
	template <typename TFloat>
	class CudaSatTanh1 : public CudaSat1<TFloat>
	{		
		using CudaSat1<TFloat>::b;
		using CudaSat1<TFloat>::k;
		using CudaSat1<TFloat>::n;
		using CudaSat1<TFloat>::m;
		using CudaSat1<TFloat>::stride;
		using CudaSat1<TFloat>::cons;
		using CudaSat1<TFloat>::gC;
		using CudaSat1<TFloat>::one;
		using CudaSat1<TFloat>::blocks;
		using CudaSat1<TFloat>::threads;
		using CudaSat1<TFloat>::Free;
		using CudaSat1<TFloat>::gMeanAm;

	protected:
		TFloat q;

	public:		

		//state types for this SAT implementation
		typedef CudaSatState1<TFloat> State;
		typedef ISatState<TFloat, CudaODEState<TFloat>> IState;
		typedef IODEState<TFloat, CudaODEState<TFloat>> IBasicState;

		CudaSatTanh1();
		~CudaSatTanh1() override;

		//get the q parameter of the Tanh
		TFloat Get_Q() const;

		//set the q parameter for the Tanh
		void Set_Q(TFloat _q);

		virtual void GetDerivatives(IBasicState& dxdt, const IBasicState& state, const TFloat time) override;
	};

}

#endif

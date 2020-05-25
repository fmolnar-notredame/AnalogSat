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


#ifndef ANALOGSAT_CUDASATSTATE_BASE_H
#define ANALOGSAT_CUDASATSTATE_BASE_H

#include "../solver/ISatState.h"
#include "CudaUtils.h"
#include <vector>

namespace analogsat
{
	//Base class for CUDA-based SAT State vectors
	//Extends the generic SatState with CUDA device copy methods
	//Provides implementations common for derived classes
	template<typename TFloat>
	class CudaSatStateImpl : public ISatState<TFloat, CudaODEState<TFloat>>
	{
	public:

		CudaODEState<TFloat>& GetState() override { return gData; }

		const CudaODEState<TFloat>& GetState() const override { return gData; }		

		operator CudaODEState<TFloat>& () override { return gData; }

		operator const CudaODEState<TFloat>& () const override { return gData; }

		void SetZero() override
		{
			CudaSafe(cudaMemsetAsync(gData, 0, sizeof(TFloat) * GetAllocSize()));
		}

		// ODE state size
		int GetAllocSize() const override { return N + M + arrayOffset + padding; }


	protected:

		//ctor with clause ordering
		CudaSatStateImpl(int _N, int _M, const std::vector<int>& _clauseOrder, int _arrayOffset, int _padding)
			: ISatState<TFloat, CudaODEState<TFloat>>(_N, _M, _clauseOrder),
			arrayOffset(_arrayOffset),
			padding(_padding)
		{ }

		//ctor without clause ordering
		CudaSatStateImpl(int _N, int _M, int _arrayOffset, int _padding)
			: ISatState<TFloat, CudaODEState<TFloat>>(_N, _M),
			arrayOffset(_arrayOffset),
			padding(_padding)
		{ }

		//copy ctor, for any base object (copy via transfer buffer)
		CudaSatStateImpl(const ISatState<TFloat, CudaODEState<TFloat>>& other, int _arrayOffset, int _padding)
			: ISatState<TFloat, CudaODEState<TFloat>>(other),
			arrayOffset(_arrayOffset),
			padding(_padding)
		{ }

		using ISatState<TFloat, CudaODEState<TFloat>>::N;
		using ISatState<TFloat, CudaODEState<TFloat>>::M;
		
		TFloat* gData;		//cuda array
		int arrayOffset;	//memory offset of N and M state vars in the cuda arrays | demand it to be set in ctor	
		int padding;		//memory padding following M state vars

		// provide implements for derived classes -- these are common for all cuda states

		void CopyPartsTo(TFloat* targetSpin, TFloat* targetAux) const override
		{
			if (targetSpin != 0) CudaSafe(cudaMemcpy(targetSpin, &gData[arrayOffset], sizeof(TFloat) * N, cudaMemcpyDeviceToHost));
			if (targetAux != 0) CudaSafe(cudaMemcpy(targetAux, &gData[N + arrayOffset], sizeof(TFloat) * M, cudaMemcpyDeviceToHost));
		}

		void CopyPartsFrom(const TFloat* sourceSpin, const TFloat* sourceAux) override
		{
			//CudaSafe(cudaMemcpy(gData, &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
			CudaSafe(cudaMemcpyAsync(&gData[arrayOffset], sourceSpin, sizeof(TFloat) * N, cudaMemcpyHostToDevice));
			CudaSafe(cudaMemcpyAsync(&gData[N + arrayOffset], sourceAux, M * sizeof(TFloat), cudaMemcpyHostToDevice));
		}

		//copy ctor implementation, to be called in derived class copy ctor after allocation
		void CopyFromGeneric(const ISatState<TFloat, CudaODEState<TFloat>>& other)
		{
			//copy state via vector
			std::vector<double> transfer(N + M);
			other.CopyPartsTo(transfer.data(), transfer.data() + N);  //no issue with ordering, since we copy the clauseOrder too.
			CopyPartsFrom(transfer.data(), transfer.data() + N);
		}

		//copy ctor implementation for cuda cases
		virtual void CopyFromCuda(const CudaSatStateImpl<TFloat>& other)
		{
			CudaSafe(cudaMemcpyAsync(&gData[arrayOffset], &other.gData[other.arrayOffset], sizeof(TFloat) * N, cudaMemcpyDeviceToDevice));
			CudaSafe(cudaMemcpyAsync(&gData[N + arrayOffset], &other.gData[N + other.arrayOffset], sizeof(TFloat) * M, cudaMemcpyDeviceToDevice));
		}

	};
}

#endif

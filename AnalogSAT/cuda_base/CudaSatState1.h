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


#ifndef ANALOGSAT_CUDASATSTATE1_H
#define ANALOGSAT_CUDASATSTATE1_H

#include "CudaSatStateImpl.h"
#include "CudaUtils.h"
#include <vector>

namespace analogsat
{
	//SAT State vector for CUDA, version 1 (used by CudaSat1)
	//offset applied for +1 variable
	//padding not applied
	template<typename TFloat>
	class CudaSatState1 : public CudaSatStateImpl<TFloat>
	{
	public:
		
		//ctor with clause ordering
		CudaSatState1(int _N, int _M, const std::vector<int>& _clauseOrder)
			: CudaSatStateImpl<TFloat>(_N, _M, _clauseOrder, 1, 0)
		{
			Allocate();
		}

		//ctor without clause ordering
		CudaSatState1(int _N, int _M)
			: CudaSatStateImpl<TFloat>(_N, _M, 1, 0)
		{
			Allocate();
		}

		//copy ctor, for any base object (copy via transfer buffer)
		CudaSatState1(const ISatState<TFloat, CudaODEState<TFloat>>& other)
			: CudaSatStateImpl<TFloat>(other, 1, 0)
		{
			Allocate();			
			CopyFromGeneric(other);
		}

		//special copy ctor for cuda objects (use cuda device to device copy)
		CudaSatState1(const CudaSatStateImpl<TFloat>& other)
			: CudaSatStateImpl<TFloat>(other)
		{
			Allocate();
			CopyFromCuda(other);
		}

		~CudaSatState1() override
		{
			SAFEDEL(gData);			
		}

	protected:

		using CudaSatStateImpl<TFloat>::N;
		using CudaSatStateImpl<TFloat>::M;
		using CudaSatStateImpl<TFloat>::gData;		//cuda array
		using CudaSatStateImpl<TFloat>::arrayOffset;	//memory offset of N and M state vars in the cuda arrays | demand it to be set in ctor
		using CudaSatStateImpl<TFloat>::padding;
		using CudaSatStateImpl<TFloat>::GetAllocSize;

		void CopyPartsFrom(const TFloat* sourceSpin, const TFloat* sourceAux) override
		{
			CudaSatStateImpl<TFloat>::CopyPartsFrom(sourceSpin, sourceAux); //base class call
			CudaSafe(cudaMemcpy(gData, &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
		}

		void CopyFromCuda(const CudaSatStateImpl<TFloat>& other) override
		{
			CudaSatStateImpl<TFloat>::CopyFromCuda(other); //base class call
			CudaSafe(cudaMemcpyAsync(gData, &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
		}

	private:

		const TFloat minusone = (TFloat)-1.0;
		size_t size;

		void Allocate() //called from ctor
		{
			size = GetAllocSize();
			CudaSafe(cudaMalloc(&gData, size * sizeof(TFloat)));
			CudaSafe(cudaMemcpyAsync(&gData[0], &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
		}
	};
}

#endif

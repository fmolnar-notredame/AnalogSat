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


#ifndef ANALOGSAT_CUDASATSTATE2_H
#define ANALOGSAT_CUDASATSTATE2_H

#include "CudaSatStateImpl.h"
#include "CudaUtils.h"
#include <vector>

namespace analogsat
{
	//SAT State vector for CUDA, version 2 (used by CudaSat2 and CudaSat3)
	//offset applied for +2 variable
	//padding applied as requested
	template<typename TFloat>
	class CudaSatState2 : public CudaSatStateImpl<TFloat>
	{
	public:
		CudaSatState2(int _N, int _M, const std::vector<int>& _clauseOrder, int _padding)
			: CudaSatStateImpl<TFloat>(_N, _M, _clauseOrder, 2, _padding)
		{
			Allocate();
		}

		//ctor without clause ordering
		CudaSatState2(int _N, int _M, int _padding)
			: CudaSatStateImpl<TFloat>(_N, _M, 2, _padding)
		{
			Allocate();
		}

		//copy ctor, for any base object (copy via transfer buffer)
		CudaSatState2(const ISatState<TFloat, CudaODEState<TFloat>>& other, int _padding)
			: CudaSatStateImpl<TFloat>(other, 2, _padding)
		{
			Allocate();
			CopyFromGeneric(other);
		}

		//special copy ctor for cuda objects (use cuda device to device copy)
		CudaSatState2(const CudaSatStateImpl<TFloat>& other, int _padding)
			: CudaSatStateImpl<TFloat>(other, _padding)
		{
			Allocate();
			CopyFromCuda(other);
		}

		~CudaSatState2() override
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
			CudaSafe(cudaMemcpyAsync(&gData[0], &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
			CudaSafe(cudaMemcpyAsync(&gData[1], &one, sizeof(TFloat), cudaMemcpyHostToDevice));
		}

		void CopyFromCuda(const CudaSatStateImpl<TFloat>& other) override
		{
			CudaSatStateImpl<TFloat>::CopyFromCuda(other); //base class call
			CudaSafe(cudaMemcpyAsync(&gData[0], &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
			CudaSafe(cudaMemcpyAsync(&gData[1], &one, sizeof(TFloat), cudaMemcpyHostToDevice));
		}

	private:

		const TFloat minusone = (TFloat)-1.0;
		const TFloat one = (TFloat)1.0;

		size_t size;

		void Allocate() //called from ctor
		{
			size = GetAllocSize();
			CudaSafe(cudaMalloc(&gData, size * sizeof(TFloat)));			
			CudaSafe(cudaMemcpyAsync(&gData[0], &minusone, sizeof(TFloat), cudaMemcpyHostToDevice));
			CudaSafe(cudaMemcpyAsync(&gData[1], &one, sizeof(TFloat), cudaMemcpyHostToDevice));
		}
		
	};
}

#endif

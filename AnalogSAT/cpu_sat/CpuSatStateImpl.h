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


#ifndef ANALOGSAT_CPUSATSTATE_H
#define ANALOGSAT_CPUSATSTATE_H

#include "../solver/ISatState.h"
#include <vector>
#include <cstring>

namespace analogsat
{
	//SAT State vector for CPU
	//Padding applied for +1 variable
	template<typename TFloat>
	class CpuSatStateImpl : public ISatState<TFloat, CpuODEState<TFloat>>
	{
	public:

		//ctor with clause ordering
		CpuSatStateImpl(int _N, int _M, const std::vector<int>& _clauseOrder)
			: ISatState<TFloat, CpuODEState<TFloat>>(_N, _M, _clauseOrder)
		{
			Allocate();
		}

		//ctor without clause ordering		
		CpuSatStateImpl(int _N, int _M)
			: ISatState<TFloat, CpuODEState<TFloat>>(_N, _M)
		{
			Allocate();
		}

		~CpuSatStateImpl() override
		{
			//data.clear(); //eh
		}

		CpuODEState<TFloat>& GetState() override { return data; }

		const CpuODEState<TFloat>& GetState() const override { return data; }		

		operator CpuODEState<TFloat>& () override { return data; }
		
		operator const CpuODEState<TFloat>& () const override { return data; }

		void SetZero() override
		{
			std::memset(data.data(), 0, sizeof(TFloat) * data.size());
		}

		// ODE state size
		int GetAllocSize() const override { return N + M + 1; }


	protected:

		using ISatState<TFloat, CpuODEState<TFloat>>::N;
		using ISatState<TFloat, CpuODEState<TFloat>>::M;

		void CopyPartsTo(TFloat* targetSpin, TFloat* targetAux) const override
		{
			if (targetSpin != 0) std::memcpy(targetSpin, data.data() + 1, sizeof(TFloat) * N);
			if (targetAux != 0) std::memcpy(targetAux, data.data() + N + 1, sizeof(TFloat) * M);
		}

		void CopyPartsFrom(const TFloat* sourceSpin, const TFloat* sourceAux) override
		{
			data[0] = minusone;
			std::memcpy(data.data() + 1, sourceSpin, sizeof(TFloat) * N);
			std::memcpy(data.data() + N + 1, sourceAux, sizeof(TFloat) * M);
		}

	private:		

		const TFloat minusone = (TFloat)-1.0;
		size_t size;
		std::vector<TFloat> data;

		void Allocate()
		{
			size = N + M + 1;
			data.resize(size);
			data[0] = minusone;
		}
	};
}

#endif

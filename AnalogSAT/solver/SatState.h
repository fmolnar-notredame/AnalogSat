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


#ifndef ANALOGSAT_SAT_WRAPPER_H
#define ANALOGSAT_SAT_WRAPPER_H

#include <vector>
#include <memory>
#include <stdexcept>
#include "ISatState.h"

namespace analogsat
{
	//Public wrapper class for ISatState pointers.
	//Use this to hold SatStates returned by ISat<>.MakeState() factory methods.
	//The whole purpose of this class is to simplify syntax for the user.
	template<typename TFloat, typename TState>
	class SatState : public ISatState<TFloat, TState>
	{
	public:
		SatState(ISatState<TFloat, TState>* _state)
			: ISatState<TFloat, TState>(_state->Get_N(), _state->Get_M()),
			state(_state)
		{ }

		~SatState() override {};

		TState& GetState() override { return state->GetState(); }

		const TState& GetState() const override { return state->GetState(); }

		operator TState& () override { return (*state).operator TState&(); }

		operator const TState& () const override { return (*state).operator const TState&(); }

		//override existing implementation: don't call protected things, just redirect everything to the pointer
		
		void CopyTo(std::vector<TFloat>& target) const override
		{
			state->CopyTo(target);
		}

		void CopyFrom(const std::vector<TFloat>& source) override
		{
			state->CopyFrom(source);
		}

		void CopyFrom(const IODEState<TFloat, TState>& other) override
		{
			state->CopyFrom(other);
		}

		void SetZero() override
		{
			state->SetZero();
		}
		
		int GetAllocSize() const override 
		{ 
			return state->GetAllocSize();
		}

	private:
		std::unique_ptr<ISatState<TFloat, TState>> state;

	protected:

		//dummy implementation, not used in this class
		void CopyPartsTo(TFloat* targetSpin, TFloat* targetAux) const override
		{ }

		//dummy implementation, not used in this class
		void CopyPartsFrom(const TFloat* sourceSpin, const TFloat* sourceAux) override
		{ }
	};

	//shortcut to CPU-based SatState wrappers
	template<typename TFloat>
	using CpuSatState = SatState<TFloat, CpuODEState<TFloat>>;

}



#endif

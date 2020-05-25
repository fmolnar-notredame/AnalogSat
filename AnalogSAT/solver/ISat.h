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


#ifndef ANALOGSAT_ISAT_H
#define ANALOGSAT_ISAT_H

#include <vector>
#include "IODE.h"
#include "../problem/SatProblem.h"
#include "ISatState.h"
#include "ISatRandom.h"

namespace analogsat
{
	//generic interface to ODEs representing a SAT solving system
	//all methods should throw exceptions until SetProblem is called first
	template<typename TFloat, typename TState>
	class ISat : public IODE<TFloat, TState>
	{
	public:
		virtual ~ISat() {};

		//set the problem for this SAT system. 
		virtual void SetProblem(const SatProblem& problem) = 0;

		//number of SAT variables
		virtual int Get_N() const = 0;

		//number of SAT clauses
		virtual int Get_M() const = 0;

		//max number of literals in a clause
		virtual int Get_K() const = 0;

		//get the sine term prefactor
		virtual TFloat Get_B() const = 0;
		
		//set the sine term prefactor
		virtual void Set_B(TFloat _b) = 0;

		//make a new sat state
		virtual ISatState<TFloat, TState>* MakeState() const = 0;

		//Side note: MakeState() from base can be overridden by ISatState<TFloat, TState>* MakeState() because ISatState<> is derived from IODEState<>.
		//Without derivation, one would hit a wall, since template classes in c++ are invariant (not covariant).

		//write random state values into the given ISatState, using the given random generator.		
		virtual void SetRandomState(ISatState<TFloat, TState>& state, ISatRandom<TFloat>& random) = 0;

		//create a new random sat state 		
		ISatState<TFloat, TState>* MakeRandomState(ISatRandom<TFloat>& random) //implementation in an interface.. well fine.
		{
			ISatState<TFloat, TState>* state = MakeState();
			SetRandomState(*state, random);
			return state;
		}

		//calculate how many clauses are violated by the given state
		virtual int GetClauseViolationCount(const ISatState<TFloat, TState>& state) const = 0;

		//calculate how many clauses are violated by the given state
		int GetClauseViolationCount(const std::vector<TFloat>& state) const
		{
			ISatState<TFloat, TState>* temp = MakeState();
			temp->CopyFrom(state);
			int count = GetClauseViolationCount(*temp);
			delete temp;
			return count;
		}

	protected:
		ISat() {};
	};

}

#endif
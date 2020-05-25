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


#ifndef ANALOGSAT_SAT_STATE_H
#define ANALOGSAT_SAT_STATE_H

#include <vector>
#include <stdexcept>
#include "IODEState.h"

namespace analogsat
{
	// Abstract base class to SAT states - derived classes should take care of actual allocation and destruction
	// Clause ordering enables proper public interfacing (copy from/to vectors),
	// without it, the state will not re-order the aux variables.
	// Note, without clauseOrder, the covertBuf will default to zero size.	
	template<typename TFloat, typename TState>
	class ISatState : public IODEState<TFloat, TState>
	{
	protected:
		//Allocates a new SatState with clause ordering		
		ISatState(int _N, int _M, const std::vector<int>& _clauseOrder)
		{
			if (_clauseOrder.size() < _M) throw std::invalid_argument("clause order is too short");

			N = _N;
			M = _M;
			clauseOrder = _clauseOrder;
			convertBuf.resize(M);
		}

		//Allocates a new SatState without clause ordering
		ISatState(int _N, int _M)
		{
			N = _N;
			M = _M;
		}

		//copy ctor -- suggest derived classes to copy the actual state via CopyFrom/CopyTo methods and a temp buffer
		ISatState(const ISatState& other)
		{
			N = other.N;
			M = other.M;
			clauseOrder = other.clauseOrder;
		}

	public:

		virtual ~ISatState() {}

		//number of SAT variables
		int Get_N() const { return N; }

		//number of SAT clauses
		int Get_M() const { return M; }

		//copies the content from this SAT state to a public vector representation
		void CopyTo(std::vector<TFloat>& target) const override
		{
			target.resize(N + M);

			if (clauseOrder.size() == 0)
			{
				CopyPartsTo(target.data(), target.data() + N);
			}
			else
			{
				std::vector<TFloat>& buf = const_cast<std::vector<TFloat>&>(convertBuf); //force-allow const correctness of the object, convertBuf is not part of the object state
				buf.resize(M);
				CopyPartsTo(target.data(), buf.data());

				//un-order the aux variables
				for (int j = 0; j < M; j++)
				{
					int idx = clauseOrder[j];
					if (idx >= 0) target[N + idx] = convertBuf[j];
				}
			}
		}

		//convert the spin states to a bool vector and copy them to target
		void CopyBooleanTo(std::vector<bool>& target) const
		{
			std::vector<TFloat>& buf = const_cast<std::vector<TFloat>&>(convertBuf);
			buf.resize(N);
			CopyPartsTo(buf.data(), 0);

			target.resize(N);
			for (int i = 0; i < N; i++) target[i] = buf[i] > 0;
		}

		//copies the public vector representation of the ODE state  into this SAT state
		void CopyFrom(const std::vector<TFloat>& source) override
		{
			if (source.size() < N + M) throw std::invalid_argument("source vector is too short");

			if (clauseOrder.size() == 0)
			{
				CopyPartsFrom(source.data(), source.data() + N);
			}
			else
			{
				//re-order the aux variables
				std::vector<TFloat>& buf = const_cast<std::vector<TFloat>&>(convertBuf); //force-allow const correctness of the object, convertBuf is not part of the public object state
				buf.resize(M);
				for (int j = 0; j < M; j++)
				{
					int idx = clauseOrder[j];
					if (idx >= 0) buf[j] = source[N + idx];
				}
				CopyPartsFrom(source.data(), convertBuf.data());
			}
		}

		//general copy assignment from any other ODE state (of equal size)
		void CopyFrom(const IODEState<TFloat, TState>& other) override
		{
			std::vector<TFloat> temp;
			other.CopyTo(temp); //if other had clause ordering, it will be applied
			CopyFrom(temp);		//if we have (different?) clause ordering, we will apply ours

			//Note,  clause ordering may even depend on each instance of the same class, so
			//reordering conversion is always needed. The only time when the clause order
			//can be assumed to be the same is copy-construction.
		}


	protected:

		int N, M;

		//copy methods to be implemented in derived classes

		//copy data from internal representation to vector data pointers
		//null can be passed, in which case that copy will be skipped
		virtual void CopyPartsTo(TFloat* targetSpin, TFloat* targetAux) const = 0;
		
		//copy from vector data to internal representation
		virtual void CopyPartsFrom(const TFloat* sourceSpin, const TFloat* sourceAux) = 0;		

	private:

		std::vector<int> clauseOrder;

		std::vector<TFloat> convertBuf; //not part of the class' public state, const_cast when used from const methods

	};

	
}



#endif

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


#ifndef ANALOGSAT_IODESTATE_H
#define ANALOGSAT_IODESTATE_H

#include <vector>

namespace analogsat
{
	//Generic interface to ODE states
	template <typename TFloat, typename TState>
	class IODEState
	{
	public:

		virtual ~IODEState(){};

		//reference
		virtual TState& GetState() = 0;	//return non-const reference to the underlying storage (encapsulation?... it's not a black box)
		
		//const reference
		virtual const TState& GetState() const = 0;	//return const reference to the underlying storage

		//size of the allocated state array
		virtual int GetAllocSize() const = 0;		
		
		//generic copy compatibility with vector<double>
		virtual void CopyTo(std::vector<TFloat>& target) const = 0;
		
		//generic copy compatibility with vector<double>
		virtual void CopyFrom(const std::vector<TFloat>& source) = 0;

		//copy assignment from another IODEState
		virtual void CopyFrom(const IODEState<TFloat, TState>& other) = 0;

		//cast operator, be able to act as the represented storage class, if needed
		virtual operator TState& () = 0;				

		//const cast operator, be able to act as the const storage class, if needed
		virtual operator const TState& () const = 0;

		virtual void SetZero() = 0;
	};

	//shortcut for CPU-based ODE state storage type
	template<typename TFloat>
	using CpuODEState = std::vector<TFloat>;

	//shortcut for CUDA-based ODE state storage type
	template<typename TFloat>
	using CudaODEState = TFloat*;

	//ODEState is not a black-box vector storage base class
	//It does not hide access to the underlying data
	//This is desirable for "native" write access and Swap operations

	//Technically, the underlying storage should only be exposed as internal,
	//and only copy methods should be public. This wrapper class is provided 
	//for SAT states, so the user does not have to deal with 
	//implementation-specific template arguments.

	//The underlying ODE state vector is allocated by ctors, freed by dtors, 
	//and all its internal parameters (subsections, like SAT's N, M, or clauseOrdering) are IMMUTABLE.
	//Copy assignment is only allowed from compatible states (else exception)
}

#endif

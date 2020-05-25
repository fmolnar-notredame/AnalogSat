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


#ifndef ANALOGSAT_IODE_H
#define ANALOGSAT_IODE_H

#include "IODEState.h"

namespace analogsat
{
	//generic interface to ordinary differential equation (ODE) systems
	template<typename TFloat, typename TState>
	class IODE
	{
	public:
		virtual ~IODE() {};
		
		//calculate the derivatives at the given state		
		virtual void GetDerivatives(IODEState<TFloat, TState>& dxdt, const IODEState<TFloat, TState>& state, const TFloat time) = 0;

		//factory method to create a state compatible with this ODE
		//overriding class should use return type covariance to match an implementation
		virtual IODEState<TFloat, TState>* MakeState() const = 0;

	protected:
		IODE() {};
	};
}
#endif


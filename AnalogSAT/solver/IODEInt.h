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


#ifndef ANALOGSAT_IODEINT_H
#define ANALOGSAT_IODEINT_H

#include "IODE.h"
#include "IODEState.h"
#include <memory>

namespace analogsat
{
	//generic interface to ODE time integrators
	template <typename TFloat, typename TState>
	class IODEInt
	{
	public:
		virtual ~IODEInt() {};

		//get ready to integrate the given ODE system
		virtual void Configure(std::shared_ptr<IODE<TFloat, TState>> odePtr) = 0;

		//take count steps of time integration
		virtual bool Step(IODEState<TFloat, TState>& oldState, IODEState<TFloat, TState>& newState, int count) = 0;

		//returns the size of the last time step (variable if the solver has adaptive time stepping)
		virtual TFloat GetLastStepSize() const = 0;

		//returns the current time variable
		virtual TFloat GetTime() const = 0;

		//returns the current count of the RHS evaluations
		virtual int GetRHSCount() const = 0;

		//returns the current count of successful steps taken
		virtual int GetStepCount() const = 0;

		//returns the number of attempted steps, including the rejected ones
		virtual int GetAttemptCount() const = 0;

		//resets time and counters
		virtual void Reset() = 0;

	protected:
		IODEInt() {};
	};
}

#endif

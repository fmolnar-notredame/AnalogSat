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


#ifndef ANALOGSAT_CPURUNGEKUTTA_H
#define	ANALOGSAT_CPURUNGEKUTTA_H

#include "../solver/IODE.h"
#include "../solver/IODEInt.h"

#include <cstring>
#include <memory>
#include <vector>

namespace analogsat
{
	//Integrate ODEs with Runge-Kutta 4/5th order solver with Adaptive time stepping
	//based on Numerical Recipes in C, 2nd ed.
	template <typename TFloat>
	class CpuRungeKutta : public IODEInt<TFloat, CpuODEState<TFloat>>
	{
	public:
		CpuRungeKutta();
		~CpuRungeKutta() override;

		typedef IODEState<TFloat, CpuODEState<TFloat>> State;
		typedef IODE<TFloat, CpuODEState<TFloat>> ODE;

		void Configure(std::shared_ptr<ODE> _odePtr) override;

		//try to carry out time steps. 
		//step size is chosen adaptively, if underflows, returns false
		bool Step(State& oldState, State& newState, int count) override;

		//returns the last timestep taken
		TFloat GetLastStepSize() const override;

		//returns the current time variable
		TFloat GetTime() const override;

		//returns the current count of the RHS evaluations
		int GetRHSCount() const override;

		//returns the numer of steps taken successfully
		int GetStepCount() const override;

		//returns the number of attempted RK4 steps
		int GetAttemptCount() const override;

		//global error tolerance
		TFloat GetEpsilon() const;
		void SetEpsilon(TFloat _eps);

		//reset to default time step and reinitialize integrator, but keep allocations
		void Reset() override;

	private:

		const TFloat TINY = TFloat(1e-20);
		const TFloat SAFETY = TFloat(0.9);
		const TFloat PGROW = TFloat(-0.2);
		const TFloat PSHRNK = TFloat(-0.25);
		const TFloat ERRCON = TFloat(1.89e-4);
		const TFloat HFACTOR = TFloat(0.1);

		const TFloat ZERO = (TFloat)0.0;
		const TFloat ONE = (TFloat)1.0;
		const TFloat FIVE = (TFloat)5.0;

		const TFloat a2 = (TFloat)0.2,
			a3 = (TFloat)0.3,
			a4 = (TFloat)0.6,
			a5 = (TFloat)1.0,
			a6 = (TFloat)0.875,
			b21 = (TFloat)0.2,
			b31 = (TFloat)(3.0 / 40.0),
			b32 = (TFloat)(9.0 / 40.0),
			b41 = (TFloat)0.3,
			b42 = (TFloat)-0.9,
			b43 = (TFloat)1.2,
			b51 = (TFloat)(-11.0 / 54.0),
			b52 = (TFloat)2.5,
			b53 = (TFloat)(-70.0 / 27.0),
			b54 = (TFloat)(35.0 / 27.0),
			b61 = (TFloat)(1631.0 / 55296.0),
			b62 = (TFloat)(175.0 / 512.0),
			b63 = (TFloat)(575.0 / 13824.0),
			b64 = (TFloat)(44275.0 / 110592.0),
			b65 = (TFloat)(253.0 / 4096.0),
			c1 = (TFloat)(37.0 / 378.0),
			c3 = (TFloat)(250.0 / 621.0),
			c4 = (TFloat)(125.0 / 594.0),
			c6 = (TFloat)(512.0 / 1771.0),
			dc5 = (TFloat)(-277.0 / 14336.0);

		const TFloat dc1 = c1 - (TFloat)(2825.0 / 27648.0),
			dc3 = c3 - (TFloat)(18575.0 / 48384.0),
			dc4 = c4 - (TFloat)(13525.0 / 55296.0),
			dc6 = c6 - (TFloat)0.25;


		void Allocate();
		void Free();

		std::shared_ptr<ODE> odePtr;	//ODEInt does not own the equations, it only holds a reference

		int n;					//number of variables

		//stats
		TFloat time;			//current time variable
		TFloat hdid, hnext;		//step size
		TFloat eps;				//global error tolerance		    
		int steps;				//number of steps taken	
		int attempts;			//number of steps attempted (including those that were rejected)
		int rhscount;			//number of times the RHS function was called

		State *temp, *dxdt, *xerr, *xscal; //state extension: error estimate, scaling,
		State *ak2, *ak3, *ak4, *ak5, *ak6; //dxdt at intermediate points

		//compute a timestep proposal
		void rkck(const std::vector<TFloat>& state, std::vector<TFloat>& xnew, TFloat h);
		bool rkqs(const std::vector<TFloat>& state, std::vector<TFloat>& xnew);
	};
}

#endif	


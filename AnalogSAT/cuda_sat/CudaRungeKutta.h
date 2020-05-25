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


#ifndef ANALOGSAT_CUDARUNGEKUTTA_H
#define ANALOGSAT_CUDARUNGEKUTTA_H

#include "cuda_runtime.h"
#include <memory>
#include <cmath>

#include "../solver/IODE.h"
#include "../solver/IODEInt.h"

namespace analogsat
{
	template <typename TFloat>
	class CudaRungeKutta : public IODEInt<TFloat, CudaODEState<TFloat>>
	{
	public:
		CudaRungeKutta();
		~CudaRungeKutta() override;

		typedef IODEState<TFloat, CudaODEState<TFloat>> State;
		typedef IODE<TFloat, CudaODEState<TFloat>> ODE;

		void Configure(std::shared_ptr<ODE> _ode) override;

		bool Step(State& oldState, State& newState, int stepsToDo) override;

		//returns the last timestep taken
		TFloat GetLastStepSize() const override;

		//returns the current time variable
		TFloat GetTime() const override;

		//returns the current count of the RHS evaluations
		int GetRHSCount() const override;

		//returns the number of successful RK4 steps
		int GetStepCount() const override;

		//returns the number of attempted RK4 steps
		int GetAttemptCount() const override;

		//resets the counters - configure calls this automatically
		void Reset() override;

		//global error tolerance
		TFloat GetEpsilon() const;
		void SetEpsilon(TFloat _eps);

	protected:
		std::shared_ptr<ODE> odePtr;	//ODEInt does not own the equations. This allows the equations (e.g. the means of computing the RHS) to change while Step()ing.

		void Allocate();
		void Free();

		//device memory pointers -- upload pointers to constant memory after allocation
		//TFloat *temp, *dxdt, *xerr, *xscal; //state extension: error estimate, scaling,
		//TFloat *ak2, *ak3, *ak4, *ak5, *ak6; //dxdt at intermediate points
		
		State *temp, *dxdt, *xerr, *xscal;
		State *ak2, *ak3, *ak4, *ak5, *ak6;

		TFloat *gerrmax1;	//reduction happens here for errmax. needs enough elements for MAX_BLOCK_DIM_SIZE
		TFloat *gerrmax2;	//reduction happens here for errmax. needs enough elements for MAX_BLOCK_DIM_SIZE
		TFloat *gtime;		//transactions of time and hdid between host and device

		dim3 blocks, threads; //kernel config

		void ConfigureConstants();

		int n; //state size

		//stats
		TFloat time;			//current time variable
		TFloat hdid, hnext;		//step sizes
		TFloat eps;				//global error tolerance		    
		int steps;				//number of steps taken (only the accepted ones)
		int attempts;			//number of steps attempted (including those that were rejected)
		int rhscount;			//number of times the RHS function was called	


		const TFloat TINY = TFloat(1e-20);
		const TFloat SAFETY = TFloat(0.9);
		const TFloat PGROW = TFloat(-0.2);
		const TFloat PSHRNK = TFloat(-0.25);
		const TFloat ERRCON = TFloat(1.89e-4);
		const TFloat HFACTOR = TFloat(0.1);

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

	};
}

#endif

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


#include "CpuRungeKutta.h"
#include "../util/utils.h"

#include <cstring>
#include <memory>
#include <cmath>

using namespace std;

namespace analogsat
{
	//Integrate ODEs with Runge-Kutta 4/5th order solver with Adaptive time stepping
	//templated for any numeric float type
	template <typename TFloat>
	CpuRungeKutta<TFloat>::CpuRungeKutta()
	{
		temp = 0;
		dxdt = 0;
		xerr = 0;
		xscal = 0;
		ak2 = 0;
		ak3 = 0;
		ak4 = 0;
		ak5 = 0;
		ak6 = 0;
		eps = (TFloat)1e-4;
	}

	template <typename TFloat>
	CpuRungeKutta<TFloat>::~CpuRungeKutta() 
	{
		Free();
	};

	template <typename TFloat>
	void CpuRungeKutta<TFloat>::Configure(shared_ptr<ODE> _odePtr)
	{
		odePtr = _odePtr;
		
		Free();
		Allocate();

		Reset();
	}

	//try to carry out time steps. 
	//step size is chosen adaptivelyl; if it underflows, it returns false
	template <typename TFloat>
	bool CpuRungeKutta<TFloat>::Step(State& oldState, State& newState, int count)
	{
		if (!odePtr) throw runtime_error("the RK4 solver has not been configured with an ODE yet");

		//dereference
		ODE& ode = *odePtr;
		vector<TFloat>& _dxdt = dxdt->GetState();
		vector<TFloat>& _xscal = xscal->GetState();
		vector<TFloat>& _oldState = oldState.GetState();

		for (int c = 0; c < count; c++)
		{
			//calculate derivatives
			ode.GetDerivatives(*dxdt, oldState, time);
			rhscount++;

			//calculate scaling (for the accuracy monitor)
			{
				for (int i = 0; i < n; i++)					
					_xscal[i] = fabs(_oldState[i]) + fabs(_dxdt[i] * hnext) + TINY;
			}

			//do timestep
			if (rkqs(oldState, newState))
			{
				steps++;
				auto& x = oldState.GetState();
				auto& y = newState.GetState();
				swap(x, y); //this does a deep-swap, so the hosting State object does not even notice
			}
			else return false; //this only happens if the auto-stepsize has underflown
		}

		return true;
	}

	//returns the last timestep taken
	template <typename TFloat>
	TFloat CpuRungeKutta<TFloat>::GetLastStepSize() const { return hdid; }

	//returns the current time variable
	template <typename TFloat>
	TFloat CpuRungeKutta<TFloat>::GetTime() const { return time; }

	//returns the current count of the RHS evaluations
	template <typename TFloat>
	int CpuRungeKutta<TFloat>::GetRHSCount() const { return rhscount; }

	//returns the numer of steps taken successfully
	template <typename TFloat>
	int CpuRungeKutta<TFloat>::GetStepCount() const { return steps; }

	//returns the number of attempted RK4 steps
	template <typename TFloat>
	int CpuRungeKutta<TFloat>::GetAttemptCount() const { return attempts; }

	//global error tolerance
	template <typename TFloat>
	TFloat CpuRungeKutta<TFloat>::GetEpsilon() const { return eps; }

	template <typename TFloat>
	void CpuRungeKutta<TFloat>::SetEpsilon(TFloat _eps)
	{
		if (_eps < (TFloat)1e-16 || _eps >(TFloat)1) throw invalid_argument("epsilon must be between 1e-16 and 1");
		eps = _eps;
	}

	template<typename TFloat>
	void CpuRungeKutta<TFloat>::Allocate()
	{
		ODE& ode = *odePtr;

		temp = ode.MakeState();
		dxdt = ode.MakeState();
		xerr = ode.MakeState();
		xscal = ode.MakeState();

		ak2 = ode.MakeState();
		ak3 = ode.MakeState();
		ak4 = ode.MakeState();
		ak5 = ode.MakeState();
		ak6 = ode.MakeState();

		n = temp->GetAllocSize();
	}

	template<typename TFloat>
	void CpuRungeKutta<TFloat>::Free()
	{
		NULLDEL(temp);		
		NULLDEL(dxdt);
		NULLDEL(xerr);
		NULLDEL(xscal);

		NULLDEL(ak2);
		NULLDEL(ak3);
		NULLDEL(ak4);
		NULLDEL(ak5);
		NULLDEL(ak6);
	}

	//reset to default time step and reinitialize integrator, but keep allocations
	template <typename TFloat>
	void CpuRungeKutta<TFloat>::Reset()
	{
		//init timestep and stuff
		hnext = (TFloat)1e-9;
		steps = 0;
		attempts = 0;
		time = (TFloat)0.0;
		rhscount = 0;
	}

	//compute a timestep proposal
	template <typename TFloat>
	void CpuRungeKutta<TFloat>::rkck(const vector<TFloat>& state, vector<TFloat>& xnew, TFloat h)
	{
		IODE<TFloat, vector<TFloat>>& ode = *odePtr; //dereference the shared_ptr only once per call
		vector<TFloat>& _temp = temp->GetState();
		vector<TFloat>& _dxdt = dxdt->GetState();
		vector<TFloat>& _xerr = xerr->GetState();
		vector<TFloat>& _ak2 = ak2->GetState();
		vector<TFloat>& _ak3 = ak3->GetState();
		vector<TFloat>& _ak4 = ak4->GetState();
		vector<TFloat>& _ak5 = ak5->GetState();
		vector<TFloat>& _ak6 = ak6->GetState();		

		for (int i = 0; i < n; i++) _temp[i] = state[i] + b21 * h * _dxdt[i];
		ode.GetDerivatives(*ak2, *temp, time + a2 * h);
		rhscount++;

		for (int i = 0; i < n; i++) _temp[i] = state[i] + h * (b31 * _dxdt[i] + b32 * _ak2[i]);
		ode.GetDerivatives(*ak3, *temp, time + a3 * h);
		rhscount++;

		for (int i = 0; i < n; i++) _temp[i] = state[i] + h * (b41 * _dxdt[i] + b42 * _ak2[i] + b43 * _ak3[i]);
		ode.GetDerivatives(*ak4, *temp, time + a4 * h);
		rhscount++;

		for (int i = 0; i < n; i++) _temp[i] = state[i] + h * (b51 * _dxdt[i] + b52 * _ak2[i] + b53 * _ak3[i] + b54 * _ak4[i]);
		ode.GetDerivatives(*ak5, *temp, time + a5 * h);
		rhscount++;

		for (int i = 0; i < n; i++) _temp[i] = state[i] + h * (b61 * _dxdt[i] + b62 * _ak2[i] + b63 * _ak3[i] + b64 * _ak4[i] + b65 * _ak5[i]);
		ode.GetDerivatives(*ak6, *temp, time + a6 * h);
		rhscount++;

		for (int i = 0; i < n; i++) xnew[i] = state[i] + h * (c1 * _dxdt[i] + c3 * _ak3[i] + c4 * _ak4[i] + c6 * _ak6[i]);
		for (int i = 0; i < n; i++) _xerr[i] = h * (dc1 * _dxdt[i] + dc3 * _ak3[i] + dc4 * _ak4[i] + dc5 * _ak5[i] + dc6 * _ak6[i]);
	}

	template <typename TFloat>
	bool CpuRungeKutta<TFloat>::rkqs(const vector<TFloat>& state, vector<TFloat>& xnew)
	{
		TFloat h = hnext;
		TFloat ermax = ZERO;

		vector<TFloat>& _xerr = xerr->GetState();
		vector<TFloat>& _xscal = xscal->GetState();

		while (true)
		{
			//try a step
			rkck(state, xnew, h);
			attempts++;

			//calculate error
			for (int i = 0; i < n; i++) _xerr[i] = fabs(_xerr[i] / _xscal[i]);

			ermax = ZERO;
			for (int i = 0; i < n; i++) ermax = fmax(ermax, _xerr[i]);
			ermax /= eps;

			//error small enough? then done
			if (ermax <= ONE) break;

			//error too large, reduce step size
			TFloat htemp = SAFETY * h * pow(ermax, PSHRNK);
			h = h >= ZERO ? fmax(htemp, HFACTOR * h) : fmin(htemp, HFACTOR * h);
			TFloat tnew = time + h;

			//time step underflow? FAIL
			if (tnew == time) return false;
		}

		//calculate new timestep (chance to grow)
		if (ermax > ERRCON)
		{
			hnext = SAFETY * h * pow(ermax, PGROW);
		}
		else
		{
			hnext = FIVE * h;
		}


		//update state
		hdid = h;
		time += hdid;

		return true;
	}


	//instantiate
	template class CpuRungeKutta<float>;
	template class CpuRungeKutta<double>;
	template class CpuRungeKutta<long double>;
}

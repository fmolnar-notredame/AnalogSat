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


#ifndef ANALOGSAT_SATSOLVER_BASE_H
#define ANALOGSAT_SATSOLVER_BASE_H

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <vector>
#include <functional>
#include <memory>

#include "ISat.h"
#include "IODEInt.h"
#include "ISatState.h"
#include "SatResult.h"
#include "ISatRandom.h"
#include "../problem/SatProblem.h"

namespace analogsat
{
	//main class for SAT solvers
	//implements all the basic SAT solving mechanisms
	//header-only
	template <typename TFloat, typename TState>
	class SatSolver
	{
	public:

		//type of callback function
		typedef std::function<bool(SatSolver<TFloat, TState>&)> CallbackFunction;

		//type of the callback's argument
		typedef SatSolver<TFloat, TState> Interface;

		//require SAT and integrator to be passed as unique_ptr, 
		//ownership will be taken to guarantee state consistency between the two objects being used for solving the SAT
		SatSolver(std::unique_ptr<ISat<TFloat, TState>> _satPtr, std::unique_ptr<IODEInt<TFloat, TState>> _odeintPtr)
		{
			satPtr = std::move(_satPtr); //convert object to shared -- to be shared with integrator -- this sharing is private, will not be given out ever.
			odeintPtr = std::move(_odeintPtr);

			//defaults
			maxTime = (TFloat)100.0;
			maxSteps = 100000;
			batchSize = 20;
			useCallback = false;
			running = false;
			Reset();
		}

		//convenience ctor (look at all that syntax garbage!) - DO NOT modify the passed pointers after this call
		SatSolver(ISat<TFloat, TState>* _satPtr, IODEInt<TFloat, TState>* _odeintPtr)
			: SatSolver(std::unique_ptr<ISat<TFloat, TState>>(_satPtr), std::unique_ptr<IODEInt<TFloat, TState>>(_odeintPtr))
		{ }

		//dtor
		virtual ~SatSolver()  { }

		//access to stats

		//get the number of ODE time steps taken so far
		int GetStepCount() const  { return steps; }

		//get the number of times the r.h.s. of the ODE was evaluated so far
		int GetRhsCount() const  { return rhsCount; }

		//get the elapsed analog time in the ODE integration
		TFloat GetElapsedTime() const  { return time; }

		//get the size of the last step taken by the ODE integrator
		TFloat GetLastStepSize() const  { return lastStepSize; }

		//get the total elapsed wall time since the start of the last Solve()
		double GetElapsedWallTime() const
		{
			if (running) //lazy: we are running and the user wants to know the walltime -> update now
			{
				//const casting because the object INTERNAL state changes with this const call (not the public one)
				clock_type& _wallnow = const_cast<clock_type&>(wallnow);
				std::chrono::duration<float>& _duration = const_cast<std::chrono::duration<float>&>(duration);
				double& _walltime = const_cast<double&>(walltime);

				_wallnow = std::chrono::high_resolution_clock::now();
				_duration = wallnow - wallstart;
				_walltime = duration.count();
			}
			return walltime;
		}

		//get the total elapsed cpu time since the start of the last Solve()
		double GetElapsedCpuTime() const
		{
			if (running) //lazy: we are running and the user wants to know the cputime -> update now
			{
				//const cast to allow INTERNAL object state change in a const method
				std::clock_t& _cpunow = const_cast<std::clock_t&>(cpunow);
				double& _cputime = const_cast<double&>(cputime);

				_cpunow = std::clock();
				_cputime = double(cpufinish - cpustart) / double(CLOCKS_PER_SEC);
			}
			return cputime;
		}

		//get the number of clauses violated by the current state
		int GetLastClauseViolationCount() const  { return lastViolationCount; }

		//set a callback function that will be invoked at every BatchSize() intervals of Solve()
		void SetCallback(typename SatSolver<TFloat, TState>::CallbackFunction _callback)  { callbackFunc = _callback; useCallback = true; }

		//remove a previously set callback function
		void ClearCallback()  { useCallback = false; }

		//set the number of steps taken between calling the callback function and checking for SAT solutions
		int GetBatchSize() const  { return batchSize; }

		//set the number of steps taken between calling the callback function and checking for SAT solutions
		void SetBatchSize(int _batchSize)
		{
			if (_batchSize < 1) throw std::invalid_argument("Batch size must be positive");
			batchSize = _batchSize;
		}

		//get the maximum analog timespan for the time integration (if exceeded, result will be SAT_MAXTIME_REACHED)
		TFloat GetMaxTime() const  { return maxTime; }

		//set the maximum analog timespan for the time integration (if exceeded, result will be SAT_MAXTIME_REACHED)
		void SetMaxTime(TFloat _maxTime)
		{
			if (_maxTime <= (TFloat)0.0) throw std::invalid_argument("Max time must be positive");
			maxTime = _maxTime;
		}

		//get the maximum number of ODE integration steps to take (if exceeded, result will be SAT_MAXSTEPS_REACHED)
		int GetMaxSteps() const  { return maxSteps; }

		//set the maximum number of ODE integration steps to take (if exceeded, result will be SAT_MAXSTEPS_REACHED)
		void SetMaxSteps(int _maxSteps)
		{
			if (_maxSteps < 1) throw std::invalid_argument("Max iterations must be positive");
			maxSteps = _maxSteps;
		}

		//run the SAT solver with the current configuration
		SatResult Solve()
		{
			return DoSolve();
		}

		//get the current SAT state (read only)
		const ISatState<TFloat, TState>& GetSatState() const  { return *oldState; }

		//set the current SAT state (copy made internally)
		void SetSatState(const ISatState<TFloat, TState>& _state)
		{
			ISat<TFloat, TState>& sat = *satPtr;
			ISatState<TFloat, TState>& state = *oldState;

			state.CopyFrom(_state);

			//update dependent stats
			lastViolationCount = sat.GetClauseViolationCount(state);
		}

		//set the current SAT state from a vector
		void SetSatState(const std::vector<TFloat>& _state)
		{
			ISat<TFloat, TState>& sat = *satPtr;
			ISatState<TFloat, TState>& state = *oldState;

			state.CopyFrom(_state);

			//update dependent stats
			lastViolationCount = sat.GetClauseViolationCount(state);
		}

		//set a random state for the current state
		void SetRandomInitialState(ISatRandom<TFloat>& random)
		{
			ISat<TFloat, TState>& sat = *satPtr;
			ISatState<TFloat, TState>& state = *oldState;

			sat.SetRandomState(state, random);

			//update dependent stats
			lastViolationCount = sat.GetClauseViolationCount(state);
		}

		//set the problem to be solved
		void SetProblem(const SatProblem& problem)
		{
			ISat<TFloat, TState>& sat = *satPtr;
			sat.SetProblem(problem);
			Reset();

			oldState.reset(sat.MakeState());
			newState.reset(sat.MakeState());

			//update dependent stats
			lastViolationCount = sat.GetClauseViolationCount(*oldState); //oldState may be allocated garbage. That's fine.
		}

	protected:

		//ODE integrator component
		std::shared_ptr<IODEInt<TFloat, TState>> odeintPtr;

		//ODE system component
		std::shared_ptr<ISat<TFloat, TState>> satPtr;

		//state variables for the ODE
		std::unique_ptr<ISatState<TFloat, TState>> oldState, newState;


		//callback 
		CallbackFunction callbackFunc;
		bool useCallback;

		//solution stats
		TFloat time;			//elapsed continuous time (cast to double from TFloat)
		int steps;				//number of ODE steps taken
		int rhsCount;			//number of RHS evaluations in the ODE
		double walltime;		//elapsed walltime
		double cputime;			//elapsed cputime
		TFloat lastStepSize;	//last hdid in RH4
		int lastViolationCount;	//number of violated clauses

		bool running;			//indicates when solving -- public funcs called via callback: go lazy when running (collect stats only if the user asks)

		//running stats
		typedef decltype(std::chrono::high_resolution_clock::now()) clock_type;
		clock_type wallstart, wallfinish, wallnow;

		std::chrono::duration<float> duration;
		std::clock_t cpustart, cpufinish, cpunow;


		TFloat maxTime;			//max continuous time
		int batchSize;			//number of iterations to issue at once (between reporting status)
		int maxSteps;			//maximum number of steps to take

		TFloat lastHostStateTime; //timestamp for cached state-dependent variables (e.g. lastViolationCount)

		//reset counters
		void Reset()
		{
			rhsCount = 0;
			walltime = 0;
			cputime = 0;
			steps = 0;
			time = (TFloat)0.0;
			lastHostStateTime = (TFloat)-1.0; //invalidate for sure
			lastViolationCount = 0;
		}

		SatResult DoSolve()
		{
			//turn on running mode
			running = true;

			//dereference only once
			ISat<TFloat, TState>& sat = *satPtr;
			IODEInt<TFloat, TState>& odeint = *odeintPtr;

			//reset stats
			Reset();

			//configure time integrator
			odeint.Configure(satPtr);

			//init timers
			wallstart = std::chrono::high_resolution_clock::now();
			cpustart = std::clock();

			SatResult result = SAT_UNKNOWN;
			if (useCallback && (bool)callbackFunc) //branch only once, not in a loop
			{
				while (time < maxTime && steps < maxSteps)
				{
					//step the ODE
					int stepsBefore = odeint.GetStepCount();
					if (!odeint.Step(*oldState, *newState, batchSize)) { result = SAT_UNDERFLOW; break; }
					int stepsTaken = odeint.GetStepCount() - stepsBefore;

					steps += stepsTaken;
					time = odeint.GetTime();
					lastViolationCount = sat.GetClauseViolationCount(*oldState);

					//update public counters
					rhsCount = odeint.GetRHSCount();
					lastStepSize = odeint.GetLastStepSize();

					//call the user function, check for user stop signal
					if (!callbackFunc(*this)) { result = SAT_ODE_INTERRUPTED; break; }

					//check for solution					
					if (lastViolationCount == 0) { result = SAT_SOLUTION_FOUND; break; }
				}
			}
			else //run without callbacks
			{
				while (time < maxTime && steps < maxSteps)
				{
					int stepsBefore = odeint.GetStepCount();
					if (!odeint.Step(*oldState, *newState, batchSize)) { result = SAT_UNDERFLOW; break; }
					int stepsTaken = odeint.GetStepCount() - stepsBefore;

					steps += stepsTaken;
					time = odeint.GetTime();
					lastViolationCount = sat.GetClauseViolationCount(*oldState);

					//check for solution
					if (lastViolationCount == 0) { result = SAT_SOLUTION_FOUND; break; }
				}
			}

			//final check 
			if (result == SAT_UNKNOWN)
			{
				if (time >= maxTime) result = SAT_MAXTIME_REACHED;
				else if (steps >= maxSteps) result = SAT_MAXITER_REACHED;
			}

			//final time
			wallfinish = std::chrono::high_resolution_clock::now();
			cpufinish = std::clock();
			duration = wallfinish - wallstart;
			walltime = duration.count();
			cputime = (cpufinish - cpustart) / CLOCKS_PER_SEC;

			//final counters
			time = odeint.GetTime();
			rhsCount = odeint.GetRHSCount();
			lastStepSize = odeint.GetLastStepSize(); //not much useful here but do it for consistency

			running = false;
			return result;
		}
	};

	//shortcut for CPU-based SatSolvers
	template<typename TFloat>
	using CpuSatSolver = SatSolver<TFloat, CpuODEState<TFloat>>;

	//shortcut for GPU-based SatSolvers
	template<typename TFloat>
	using CudaSatSolver = SatSolver<TFloat, CudaODEState<TFloat>>;
}

#endif

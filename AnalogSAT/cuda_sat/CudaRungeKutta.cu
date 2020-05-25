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
//along with this program. If not, see <http://www.gnu.org/licenses/>.;


#include <algorithm>
#include "CudaRungeKutta.h"
#include "../cuda_base/CudaUtils.h"
#include "../cuda_base/CudaReduce.h"
#include "../util/utils.h"

using namespace std;

namespace analogsat
{
	// RK45 constants and gpu array pointers
	template <typename TFloat>
	struct __align__(8) devconst
	{
		TFloat GTINY, GSAFETY, GPGROW, GPSHRNK, GERRCON, GHFACTOR, GEPS;
		TFloat A2, A3, A4, A5, A6, B21, B31, B32, B41, B42, B43, B51, B52, B53, B54, B61, B62, B63, B64, B65;
		TFloat C1, C3, C4, C6, DC5, DC1, DC3, DC4, DC6;

		TFloat *TEMP, *DXDT, *XERR, *XSCAL; //state extension: error estimate, scaling,
		TFloat *AK2, *AK3, *AK4, *AK5, *AK6; //dxdt at intermediate points
		TFloat *ERRMAX;	//reduction happens here for errmax. needs enough elements for MAX_BLOCK_DIM_SIZE
		TFloat *TIME;		//transactions of time and hdid between host and device
	};

	__constant__ devconst<float> fconst;
	__constant__ devconst<double> dconst;

	__constant__ int STATESIZE;


	template <typename TFloat>
	__device__ __inline__ devconst<TFloat>& GetConsODE();
	template <> __device__ __inline__ devconst<double>& GetConsODE<double>() { return dconst; }
	template <> __device__ __inline__ devconst<float>& GetConsODE<float>() { return fconst; }


	//RK45  kernels for each phase

	template <typename TFloat>
	__global__ void Calculate_ak2(TFloat* state, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < STATESIZE)
		{
			cons.TEMP[i] = state[i] + cons.B21 * h * cons.DXDT[i];
			//cons.AK2[i] = 0;
		}
	}

	template <typename TFloat>
	__global__ void Calculate_ak3(TFloat* state, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < STATESIZE)
		{
			cons.TEMP[i] = state[i] + h * (cons.B31 * cons.DXDT[i] + cons.B32 * cons.AK2[i]);
			//cons.AK3[i] = 0;
		}
	}


	template <typename TFloat>
	__global__ void Calculate_ak4(TFloat* state, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < STATESIZE)
		{
			cons.TEMP[i] = state[i] + h * (cons.B41 * cons.DXDT[i] + cons.B42 * cons.AK2[i] + cons.B43 * cons.AK3[i]);
			//cons.AK4[i] = 0;
		}
	}


	template <typename TFloat>
	__global__ void Calculate_ak5(TFloat* state, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < STATESIZE)
		{
			cons.TEMP[i] = state[i] + h * (cons.B51 * cons.DXDT[i] + cons.B52 * cons.AK2[i] + cons.B53 * cons.AK3[i] + cons.B54 * cons.AK4[i]);
			//cons.AK5[i] = 0;
		}
	}


	template <typename TFloat>
	__global__ void Calculate_ak6(TFloat* state, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < STATESIZE)
		{
			cons.TEMP[i] = state[i] + h * (cons.B61 * cons.DXDT[i] + cons.B62 * cons.AK2[i] + cons.B63 * cons.AK3[i] + cons.B64 * cons.AK4[i] + cons.B65 * cons.AK5[i]);
			//cons.AK6[i] = 0;
		}
	}


	template <typename TFloat>
	__global__ void Calculate_New(TFloat* state, TFloat *xnew, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		TFloat _state, _dxdt, _ak3, _ak4, _ak5, _ak6, _xerr;
		if (i < STATESIZE)
		{
			_state = state[i];
			_dxdt = cons.DXDT[i];
			_ak3 = cons.AK3[i];
			_ak4 = cons.AK4[i];
			_ak6 = cons.AK6[i];

			xnew[i] = _state + h * (cons.C1 * _dxdt + cons.C3 * _ak3 + cons.C4 * _ak4 + cons.C6 * _ak6);

			_ak5 = cons.AK5[i];
			_xerr = h * (cons.DC1 * _dxdt + cons.DC3 * _ak3 + cons.DC4 * _ak4 + cons.DC5 * _ak5 + cons.DC6 * _ak6);

			cons.TEMP[i] = devabs(_xerr / cons.XSCAL[i]);
		}
		//if (i == 0) cons.ERRMAX[0] = 0; //reset for atomic adds

	}


	template <typename TFloat>
	__global__ void Calculate_Scaling(TFloat* state, TFloat h)
	{
		devconst<TFloat>& cons = GetConsODE<TFloat>();
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < STATESIZE)
		{
			//cons.XSCAL[i] = devabs(state[i] + cons.DXDT[i] * h) + cons.GTINY;
			cons.XSCAL[i] = devabs(state[i]) + devabs(cons.DXDT[i] * h) + cons.GTINY;
		}
	}

	// ODEInt implementation
	template<typename TFloat>
	CudaRungeKutta<TFloat>::CudaRungeKutta()
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
		gerrmax1 = 0;
		gerrmax2 = 0;
		gtime = 0;
		eps = (TFloat)1e-4;
	}

	template<typename TFloat>
	void CudaRungeKutta<TFloat>::Configure(std::shared_ptr<ODE> _ode)
	{
		odePtr = _ode;

		Free();
		Allocate();
		ConfigureConstants();

		int blocksize = 128;
		threads = dim3(blocksize);
		blocks = dim3((int)ceil((double)n / blocksize));

		Reset();
	}


	template<typename TFloat>
	CudaRungeKutta<TFloat>::~CudaRungeKutta()
	{
		Free();
	}

	template<typename TFloat>
	void UploadConstants(devconst<TFloat>& cons);

	template<> void UploadConstants<double>(devconst<double>& cons)
	{
		CudaSafe(cudaMemcpyToSymbol(dconst, &cons, sizeof(cons)));
	}

	template<> void UploadConstants<float>(devconst<float>& cons)
	{
		CudaSafe(cudaMemcpyToSymbol(fconst, &cons, sizeof(cons)));
	}


	template<typename TFloat>
	void CudaRungeKutta<TFloat>::ConfigureConstants() //call after Allocate() !!!
	{
		ODE& ode = *odePtr;
		State& tmp = *temp;
		
		n = tmp.GetAllocSize();
		CudaSafe(cudaMemcpyToSymbol(STATESIZE, &n, sizeof(int)));

		devconst<TFloat> cons;
		cons.A2 = a2;
		cons.A3 = a3;
		cons.A4 = a4;
		cons.A5 = a5;
		cons.A6 = a6;

		cons.B21 = b21;
		cons.B31 = b31;
		cons.B32 = b32;
		cons.B41 = b41;
		cons.B42 = b42;
		cons.B43 = b43;
		cons.B51 = b51;
		cons.B52 = b52;
		cons.B53 = b53;
		cons.B54 = b54;
		cons.B61 = b61;
		cons.B62 = b62;
		cons.B63 = b63;
		cons.B64 = b64;
		cons.B65 = b65;

		cons.C1 = c1;
		cons.C3 = c3;
		cons.C4 = c4;
		cons.C6 = c6;

		cons.DC1 = dc1;
		cons.DC3 = dc3;
		cons.DC4 = dc4;
		cons.DC5 = dc5;
		cons.DC6 = dc6;

		cons.GTINY = TINY;
		cons.GSAFETY = SAFETY;
		cons.GPGROW = PGROW;
		cons.GPSHRNK = PSHRNK;
		cons.GERRCON = ERRCON;
		cons.GHFACTOR = HFACTOR;
		cons.GEPS = eps;

		cons.TEMP = temp->GetState();
		cons.DXDT = dxdt->GetState();
		cons.XERR = xerr->GetState();
		cons.XSCAL = xscal->GetState();
		cons.AK2 = ak2->GetState();
		cons.AK3 = ak3->GetState();
		cons.AK4 = ak4->GetState();
		cons.AK5 = ak5->GetState();
		cons.AK6 = ak6->GetState();
		cons.ERRMAX = gerrmax1;
		cons.TIME = gtime;

		UploadConstants<TFloat>(cons);
	}

	template<typename TFloat>
	void CudaRungeKutta<TFloat>::Reset()
	{
		//init timestep and stuff
		hnext = (TFloat)1e-9;
		steps = 0;
		attempts = 0;
		time = (TFloat)0.0;
		rhscount = 0;
	}

	//returns the last timestep taken
	template<typename TFloat>
	TFloat CudaRungeKutta<TFloat>::GetLastStepSize() const { return hdid; }

	//returns the current time variable
	template<typename TFloat>
	TFloat CudaRungeKutta<TFloat>::GetTime() const { return time; }

	//returns the current count of the RHS evaluations
	template<typename TFloat>
	int CudaRungeKutta<TFloat>::GetRHSCount() const { return rhscount; }

	//returns the number of successful RK4 steps
	template<typename TFloat>
	int CudaRungeKutta<TFloat>::GetStepCount() const { return steps; }

	//returns the number of attempted RK4 steps
	template<typename TFloat>
	int CudaRungeKutta<TFloat>::GetAttemptCount() const { return attempts; }

	template<typename TFloat>
	TFloat CudaRungeKutta<TFloat>::GetEpsilon() const { return eps; }

	template<typename TFloat>
	void CudaRungeKutta<TFloat>::SetEpsilon(TFloat _eps)
	{
		if (_eps < (TFloat)1e-16 || _eps >(TFloat)1) throw invalid_argument("epsilon must be between 1e-16 and 1");
		eps = _eps;
		if (odePtr) ConfigureConstants();
	}


	template<typename TFloat>
	void CudaRungeKutta<TFloat>::Allocate()
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

		CudaSafe(cudaMalloc(&gerrmax1, sizeof(TFloat) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gerrmax2, sizeof(TFloat) * MAX_BLOCK_DIM_SIZE));
		CudaSafe(cudaMalloc(&gtime, sizeof(TFloat) * 4));

		//ensure dxdt is set to zero before the first Step()
		//CudaSafe(cudaMemset(dxdt->GetState(), 0, sizeof(TFloat) * dxdt->GetStateSize()));
	}

	template<typename TFloat>
	void CudaRungeKutta<TFloat>::Free()
	{
		NULLDEL(temp);
		NULLDEL(temp);
		NULLDEL(dxdt);
		NULLDEL(xerr);
		NULLDEL(xscal);

		NULLDEL(ak2);
		NULLDEL(ak3);
		NULLDEL(ak4);
		NULLDEL(ak5);
		NULLDEL(ak6);

		SAFEDEL(gerrmax1);
		SAFEDEL(gerrmax2);
		SAFEDEL(gtime);
	}


	template<typename TFloat>
	bool CudaRungeKutta<TFloat>::Step(State& oldState, State& newState, int stepsToDo)
	{
		if (!odePtr) throw runtime_error("the RK4 solver has not been configured with an ODE yet");

		TFloat h;
		TFloat ermax;
		ODE& ode = *odePtr;

		for (int it = 0; it < stepsToDo; it++)
		{
			//initialize the RHS to zero
			//ZeroRHS<TFloat> KERNEL_ARGS2(blocks, threads)();
			//dxdt->SetZero();

			//launch the passed RHS kernel
			ode.GetDerivatives(*dxdt, oldState, time);
			rhscount++;

			//calculate scaling (for the accuracy monitor)
			Calculate_Scaling<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, hnext);

			// ----------------- rkqs --------------------------------------------------
			h = hnext;
			while (true)
			{
				// ------------------ rckk -------------------------------------------------
				Calculate_ak2<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, h);
				ode.GetDerivatives(*ak2, *temp, time + a2 * h);
				rhscount++;

				Calculate_ak3<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, h);
				ode.GetDerivatives(*ak3, *temp, time + a3 * h);
				rhscount++;

				Calculate_ak4<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, h);
				ode.GetDerivatives(*ak4, *temp, time + a4 * h);
				rhscount++;

				Calculate_ak5<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, h);
				ode.GetDerivatives(*ak5, *temp, time + a5 * h);
				rhscount++;

				Calculate_ak6<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, h);
				ode.GetDerivatives(*ak6, *temp, time + a6 * h);
				rhscount++;

				Calculate_New<TFloat> KERNEL_ARGS2(blocks, threads)(oldState, newState, h);

				attempts++;
				// ---------------------------------------------------------------------------

				//calculate error, reduction!  temp = abs(xerr / xscal)			
				CudaReduceMax(n, temp->GetState(), gerrmax1, gerrmax2);

				//download the result (single value)
				CudaSafe(cudaMemcpy(&ermax, gerrmax1, sizeof(TFloat), cudaMemcpyDeviceToHost));

				//scale the error
				ermax /= eps;

				//if (steps < 10) printf("  ermax: %e\n", ermax);

				//error small enough? then done
				if (ermax <= (TFloat)1.0) break;

				//error too large, reduce step size
				TFloat htemp = SAFETY * h * pow(ermax, PSHRNK);  //-0.25			
				h = h >= (TFloat)0.0 ? std::fmax(htemp, HFACTOR * h) : std::fmin(htemp, HFACTOR * h);
				TFloat tnew = time + h;

				//printf("  shr %12.11e\tnew: %12.11e\n", pow(ermax, PSHRNK), h);

				//time step underflow? FAIL
				if (tnew == time) { h = (TFloat)-1.0; break; }
			}

			//IF step was ok, calculate new stepsize (chance to grow)
			if (h > (TFloat)0.0)
			{
				if (ermax > ERRCON) //grow a little
				{
					hnext = SAFETY * h * pow(ermax, PGROW); //-0.2
					//printf("  gro %12.11e\tnew: %12.11e\n", pow(ermax, PGROW), hnext);
				}
				else //grow much
				{
					hnext = (TFloat)5.0 * h;
					//printf("  gro %12.11e\tnew: %12.11e\n", pow(ermax, PGROW), hnext);
				}

				//update state			
				time += h;
				hdid = h;

				steps++;

				//swap old-new locally
				CudaODEState<TFloat> tempState = oldState.GetState();
				oldState.GetState() = newState.GetState();
				newState.GetState() = tempState;

			}
			else //negative hnext indicates underflow, fail
			{
				//printf("  UNDERFLOW\n");
				hnext = h;
				break;
			}

			//printf("    t = %lf\n", time);
			// ----------------------end rkqs --------------------------------

		}

		return hnext > (TFloat)0.0;
	}

	//instantiate classes
	template class CudaRungeKutta<double>;
	template class CudaRungeKutta<float>;

}

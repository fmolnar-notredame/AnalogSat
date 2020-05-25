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


#ifndef ANALOGSAT_CUDARANDOM_H
#define ANALOGSAT_CUDARANDOM_H

#include "../solver/ISatRandom.h"
#include <random>
#include <chrono>
#include "curand.h"

namespace analogsat
{
	//Random number generation, implements ISatRandom, using CUDA
	template <typename TFloat> 
	class CudaRandom : public ISatRandom<TFloat>
	{
	private:
		curandGenerator_t prngGPU; //handle to CUDA random generator

	public:

		//initialize with a random seed (based on current time)
		CudaRandom();

		//initialize with the given random seed
		CudaRandom(unsigned int seed);

		~CudaRandom() override;

		void GenerateUniform(TFloat* addr, int length) override;
	};

}

#endif
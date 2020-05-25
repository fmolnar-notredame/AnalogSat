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


#include "CudaRandom.h"
#include "CudaUtils.h"
#include <chrono>

namespace analogsat
{
	//initialize with a random seed (based on current time)
	template <typename TFloat>
	CudaRandom<TFloat>::CudaRandom()
	{		
		CurandSafe(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
		CurandSafe(curandSetPseudoRandomGeneratorSeed(prngGPU, (unsigned long)std::chrono::system_clock::now().time_since_epoch().count()));		
	}

	//initialize with the given random seed
	template <typename TFloat>
	CudaRandom<TFloat>::CudaRandom(unsigned int seed)
	{		
		CurandSafe(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
		CurandSafe(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));		
	}

	template <typename TFloat>
	CudaRandom<TFloat>::~CudaRandom()
	{		
		curandDestroyGenerator(prngGPU);
	}

	template <typename TFloat>
	void CudaRandom<TFloat>::GenerateUniform(TFloat* addr, int length)
	{ }

	template<>
	void CudaRandom<float>::GenerateUniform(float* addr, int length)
	{		
		CurandSafe(curandGenerateUniform(prngGPU, addr, length));
	}

	template<>
	void CudaRandom<double>::GenerateUniform(double* addr, int length)
	{		
		CurandSafe(curandGenerateUniformDouble(prngGPU, addr, length));
	}

	//instantiate
	template class CudaRandom<float>;
	template class CudaRandom<double>;

}
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


#include "CpuRandom.h"
#include <cmath>

using namespace std;

namespace analogsat
{
	template<typename TFloat>
	CpuRandom<TFloat>::CpuRandom()
	{
		generator.seed((unsigned long)std::chrono::system_clock::now().time_since_epoch().count());
	}
	
	template<typename TFloat>
	CpuRandom<TFloat>::CpuRandom(unsigned int seed)
	{
		generator.seed(seed);
	}

	template<typename TFloat>
	CpuRandom<TFloat>::~CpuRandom() {}


	template<typename TFloat>
	void CpuRandom<TFloat>::GenerateUniform(TFloat* addr, int length)
	{
		if (length == 1) *addr = rand(generator);
		else
		{
			for (int i = 0; i < length; i++)
			{
				addr[i] = rand(generator);
			}
		}
	}

	//instantiate
	template class CpuRandom<float>;
	template class CpuRandom<double>;
}
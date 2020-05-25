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


#ifndef ANALOGSAT_SATRANDOM_H
#define ANALOGSAT_SATRANDOM_H

#include "../solver/ISatRandom.h"
#include <random>
#include <chrono>

namespace analogsat
{
	//CPU implementation for ISatRandom
	template<typename TFloat>
	class CpuRandom : public ISatRandom<TFloat>
	{
	private:
		std::default_random_engine generator;
		std::uniform_real_distribution<TFloat> rand;

	public:
		
		//initialize with a random seed (based on current time)
		CpuRandom();

		//initialize with the given random seed
		CpuRandom(unsigned int seed);
		
		~CpuRandom() override;

		void GenerateUniform(TFloat* addr, int length) override;
	};
}

#endif
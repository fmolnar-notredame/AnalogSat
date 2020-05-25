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


#ifndef ANALOGSAT_RANDOMSATMAKER_H
#define ANALOGSAT_RANDOMSATMAKER_H

#include "ISatMaker.h"
#include <random>

namespace analogsat
{
	//random SAT problem generator
	//SAT solution is not guaranteed
	//clauses are not repeated and they do not have the same variable twice
	class RandomSatMaker : public ISatMaker
	{
	public:
		RandomSatMaker();
		RandomSatMaker(unsigned long seed);

		virtual ~RandomSatMaker() override;

		void Set_K(int _k);
		int Get_K() const;

		void Set_M(int _m);
		int Get_M() const;

		void Set_N(int _n);
		int Get_N() const;

		//make the configured SAT problem
		virtual void MakeSatProblem(SatProblem& problem) override;

	protected:

		int k, m, n;
		std::vector<int> clause; //temp vector, where new ones are made
		
		std::default_random_engine generator;
		std::uniform_real_distribution<double> rand;
		
		virtual void SetDefaults();
		
		void MakeRandomClause();
		bool HasRepeatedVariables();
	};
}

#endif

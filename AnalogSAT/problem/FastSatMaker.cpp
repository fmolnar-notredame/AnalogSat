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


#include <chrono>
#include "FastSatMaker.h"
#include "../util/VectorHash.h"
#include "../util/utils.h"

using namespace std;

namespace analogsat
{
	FastSatMaker::FastSatMaker()
	{
		generator.seed((unsigned long)std::chrono::system_clock::now().time_since_epoch().count());
		SetDefaults();
	}

	FastSatMaker::FastSatMaker(unsigned long seed)
	{
		generator.seed(seed);
		SetDefaults();
	}

	void FastSatMaker::SetDefaults()
	{
		//defaults
		k = 3;
		n = 100;
		m = 325;
		clause.resize(k);
	}

	FastSatMaker::~FastSatMaker() {}

	void FastSatMaker::Set_K(int _k) { k = _k; clause.resize(k); }
	int FastSatMaker::Get_K() const { return k; }

	void FastSatMaker::Set_N(int _n) { n = _n; }
	int FastSatMaker::Get_N() const { return n; }

	void FastSatMaker::Set_M(int _m) { m = _m; }
	int FastSatMaker::Get_M() const { return m; }

	void FastSatMaker::MakeSatProblem(SatProblem& problem)
	{
		//local storeage
		vector<int> literals;
		vector<int> lens;
		literals.resize(k * m, 0);
		lens.resize(m, k);
		int* ptr = literals.data();

		//make clauses
		for (int i = 0; i < m; i++)
		{
			//make a non-repeated clause (keep making new ones until we make a clause that has not been made yet)
			MakeRandomClause();

			//copy clause into storage
			for (int j = 0; j < k; j++) *(ptr + i * k + j) = clause[j];
		}

		//build problem
		problem.Clear();
		problem.AddClauses(literals, lens);
	}

	//check for repeated variables, assuming the clause has already been sorted
	bool FastSatMaker::HasRepeatedVariables()
	{
		for (int i = 0; i < k - 1; i++)
			if (clause[i] == clause[i + 1]) return true;
		return false;
	}

	//make random clauses, make them ready for hashing
	void FastSatMaker::MakeRandomClause()
	{
		//create a valid variable selection without repetitions
		do
		{
			//choose random variables
			for (int i = 0; i < k; i++) clause[i] = ((int)floor(rand(generator) * n) + 1);

			//ensure sortedness
			LocalSort(clause, 0, k);

		} while (HasRepeatedVariables());

		//randomly assign signs
		for (int i = 0; i < k; i++) clause[i] *= rand(generator) < 0.5 ? 1 : -1;
	}

}
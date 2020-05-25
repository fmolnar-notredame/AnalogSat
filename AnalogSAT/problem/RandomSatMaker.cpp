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
#include "RandomSatMaker.h"
#include "../util/VectorHash.h"
#include "../util/utils.h"

using namespace std;

namespace analogsat
{
	RandomSatMaker::RandomSatMaker()
	{
		generator.seed((unsigned long)std::chrono::system_clock::now().time_since_epoch().count());
		SetDefaults();
	}

	RandomSatMaker::RandomSatMaker(unsigned long seed)
	{
		generator.seed(seed);
		SetDefaults();
	}

	void RandomSatMaker::SetDefaults()
	{
		//defaults
		k = 3;
		n = 100;
		m = 325;
		clause.resize(k);
	}

	RandomSatMaker::~RandomSatMaker() {}

	void RandomSatMaker::Set_K(int _k) { k = _k; clause.resize(k); }
	int RandomSatMaker::Get_K() const { return k; }

	void RandomSatMaker::Set_N(int _n) { n = _n; }
	int RandomSatMaker::Get_N() const { return n; }

	void RandomSatMaker::Set_M(int _m) { m = _m; }
	int RandomSatMaker::Get_M() const { return m; }

	void RandomSatMaker::MakeSatProblem(SatProblem& problem)
	{
		//create the custom hash tools: hash of the clauses ensures uniqueness
		VectorHash<int> hasher;
		VectorCompare<int> comparator;
		std::unordered_set<VectorWrap<int>, VectorHash<int>, VectorCompare<int>> clauseSet(32, hasher, comparator);
		
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
			while (true)
			{
				MakeRandomClause();

				//copy clause into storage
				for (int j = 0; j < k; j++) *(ptr + i * k + j) = clause[j];

				//wrapper (computes hash)
				VectorWrap<int> vec(ptr + i * k, ptr + (i + 1) * k);

				//check for uniqueness
				if (clauseSet.find(vec) == clauseSet.end())
				{
					clauseSet.insert(vec);
					break;
				}
			}
		}

		//build problem
		problem.Clear();
		problem.AddClauses(literals, lens);		
	}

	//check for repeated variables, assuming the clause has already been sorted
	bool RandomSatMaker::HasRepeatedVariables()
	{
		for (int i = 0; i < k - 1; i++)
			if (clause[i] == clause[i + 1]) return true;
		return false;
	}

	//make random clauses, make them ready for hashing
	void RandomSatMaker::MakeRandomClause()
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
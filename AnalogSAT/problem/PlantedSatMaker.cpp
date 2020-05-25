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
#include "PlantedSatMaker.h"
#include "../util/VectorHash.h"
#include "../util/utils.h"

using namespace std;

namespace analogsat
{
	PlantedSatMaker::PlantedSatMaker() : RandomSatMaker()
	{ }

	PlantedSatMaker::PlantedSatMaker(unsigned long seed) : RandomSatMaker(seed)
	{ }

	PlantedSatMaker::~PlantedSatMaker() {}

	void PlantedSatMaker::MakeSatProblem(SatProblem& problem)
	{
		//create the custom hash tools: hash of the clauses ensures uniqueness
		VectorHash<int> hasher;
		VectorCompare<int> comparator;
		std::unordered_set<VectorWrap<int>, VectorHash<int>, VectorCompare<int>> clauseSet(32, hasher, comparator);

		//local storage
		vector<int> literals;
		vector<int> lens;
		literals.resize(k * m, 0);
		lens.resize(m, k);
		int* ptr = literals.data();

		//create the planted solution
		planted.resize(n);
		for (int i = 0; i < n; i++) planted[i] = (rand(generator) < 0.5);

		//make clauses
		for (int i = 0; i < m; i++)
		{
			//make a non-repeated clause
			while (true)
			{
				//make it
				MakeRandomClause();

				//ensure that the current clause is SAT under the planted solution
				if (!Satisfied()) continue;

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

	std::vector<bool> PlantedSatMaker::GetPlantedSolution()
	{
		return planted;
	}

	//check if the current clause satisfies the planted solution
	bool PlantedSatMaker::Satisfied()
	{
		//does it satisfy the desired solution?
		bool satisfied = false;
		for (int i = 0; i < k; i++)
		{
			int item = clause[i];
			if (item < 0) satisfied |= (!planted[-item - 1]);
			else satisfied |= planted[item - 1];
		}
		return satisfied;
	}

}

//This file is part of AnalogSAT Frontend
//Copyright(C) 2019 Ferenc Molnar
//
//AnalogSAT Frontend is free software: you can redistribute it and / or modify
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


#include "minisat_adapter.h"

namespace analogsat
{

	MinisatAdapter::MinisatAdapter(const SatProblem& problem)
	{
		//construct the CNF content as a single int vector

		int N = problem.Get_N();
		int M = problem.Get_M();
		content.reserve(M * (problem.Get_K() + 1) + 2); //more than enough for mixed-SAT (exact for K-SAT)

		//first two numbers: N and M
		content.push_back(N);
		content.push_back(M);

		//literals, clauses separated by zeros
		for (int j = 0; j < M; j++)
		{
			const auto& clause = problem.GetClause(j);
			for (int i : clause) content.push_back(i);
			content.push_back(0);
		}

		it = content.begin();
	}

	bool MinisatAdapter::Finished()
	{
		return it == content.end();
	}

	int MinisatAdapter::NextInt()
	{
		int val = *it;
		it++;
		return val;
	}
}



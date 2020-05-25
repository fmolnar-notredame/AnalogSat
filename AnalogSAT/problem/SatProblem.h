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


#ifndef ANALOGSAT_SATPROBLEM_H
#define ANALOGSAT_SATPROBLEM_H

#include <vector>
#include <random>
#include <unordered_set>

#include "Clause.h"
#include "ClauseHelper.h"
#include "SatLiteral.h"

namespace analogsat
{
	//stores a set of CNF clauses.
	class SatProblem
	{
	private:
		std::vector<int> storage;
		std::vector<Clause> clauses;

		int n, m, k;

	public:

		//default ctor
		SatProblem();

		//Add a new clause to the problem.
		void AddClause(const std::vector<int>& clause);

		//Add a set of clauses to the problem
		void AddClauses(const std::vector<int>& literals, const std::vector<int>& lengths);

		//add a set of clauses, old format with vector of vectors
		void AddClauses(const std::vector<std::vector<int>>& clauses);

		//get a copy of all literals and clauses
		void GetAllClauses(std::vector<int>& literals, std::vector<int>& lengths) const;

		//Removes all clauses
		void Clear();

		//high level access to clauses by literal iterators
		//usage: for (int literal : problem.GetClause(i)) ...
		SatLiterals GetClause(int index) const;
				
		//low level access to clauses
		const int* GetLiterals() const;
		int GetClauseStart(int index) const;
		int GetClauseLength(int index) const;

		//stats
		int Get_K() const;
		int Get_N() const;
		int Get_M() const;
	};

	
}

#endif
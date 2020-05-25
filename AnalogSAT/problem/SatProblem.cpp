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


#include "SatProblem.h"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include "../util/utils.h"


using namespace std;

namespace analogsat
{
	SatProblem::SatProblem()
	{
		n = m = k = 0;
		storage.reserve(100);		
	}

	void SatProblem::AddClause(const std::vector<int>& clause)
	{
		//copy literals to storage while filtering zeros and calculating max variable indices		
		int start1 = (int)storage.size();
		int len1 = 0;
		int n_new = n;
		for (int i : clause)
		{
			if (i == 0) continue;
			storage.push_back(i);
			n_new = std::max(n_new, abs(i));
			len1++;
		}

		if (len1 == 0) return;

		//create the wrapper clause object
		Clause c(start1, len1);

		//add clause, update stats
		clauses.push_back(c);
		k = std::max(k, len1);
		n = n_new;
		m++;
	}

	void SatProblem::AddClauses(const std::vector<int>& literals, const std::vector<int>& lengths)
	{
		storage.reserve(storage.size() + literals.size());
		int count = (int)lengths.size();
		int offset = 0;
		for (int j = 0; j < count; j++)
		{
			int len = lengths[j];				//input literals to process
			int start1 = (int)storage.size();	//where to put them
			int len1 = 0;						//how many were actually put
			int n_new = n;
			for (int i = 0; i < len; i++)
			{
				int lit = literals[offset++];
				if (lit == 0) continue;
				storage.push_back(lit);
				n_new = std::max(n_new, abs(lit));
				len1++;
			}

			if (len1 == 0) continue;

			//make clause object
			Clause c(start1, len1);

			//add clause, update stats
			clauses.push_back(c);
			k = std::max(k, len1);
			n = n_new;
			m++;
		}
	}

	void SatProblem::AddClauses(const std::vector<std::vector<int>>& clauses)
	{
		int len = 0;
		for (auto& clause : clauses) len += (int)clause.size();
		storage.reserve(storage.size() + len);

		for (auto& clause : clauses) AddClause(clause);
	}

	void SatProblem::Clear()
	{
		clauses.clear();
		n = m = k = 0;
	}

	SatLiterals SatProblem::GetClause(int index) const
	{
		//if (index < 0 || index >= m) throw invalid_argument("clause index out of range");
		return SatLiterals(storage, clauses[index].GetStart(), clauses[index].GetLength());
	}

	void SatProblem::GetAllClauses(std::vector<int>& literals, std::vector<int>& lengths) const
	{
		literals = storage; //copy-assign
		lengths.clear();
		lengths.reserve(m);
		for (int i = 0; i < m; i++) lengths.push_back(clauses[i].GetLength());
	}

	const int* SatProblem::GetLiterals() const
	{
		return storage.data();
	}

	int SatProblem::GetClauseStart(int index) const
	{
		return clauses[index].GetStart();
	}

	int SatProblem::GetClauseLength(int index) const
	{
		return clauses[index].GetLength();
	}

	int SatProblem::Get_K() const { return k; }
	int SatProblem::Get_N() const { return n; }
	int SatProblem::Get_M() const { return m; }
		
}

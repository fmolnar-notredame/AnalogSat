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


#ifndef ANALOGSAT_MINISAT_ADAPTOR_H
#define ANALOGSAT_MINISAT_ADAPTOR_H

#include "analogsat.h"
#include "minisat.h"

namespace analogsat
{
	// implement the in-memory CNF provider for Minisat
	// creates a copy of the given SatProblem in memory and feeds it to MiniSat
	class MinisatAdapter : public Minisat::MemoryCNF
	{
	private:
		std::vector<int> content;
		std::vector<int>::iterator it;

	public:
		MinisatAdapter(const SatProblem& problem);

		bool Finished() override;

		int NextInt() override;
	};
}

#endif
//This file is part of Minisat for AnalogSAT
//Copyright(C) 2019 Ferenc Molnar
//
//Minisat for AnalogSAT is free software: you can redistribute it and / or modify
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

// FM 2019:
// Adding a class to provide a DIMACS CNF in memory, without reading files
// This is an abstract base class, the user should implement the means to 
// pass the CNF content.

#ifndef MINISAT_MEMCNF_H
#define MINISAT_MEMCNF_H

namespace Minisat 
{
	class MemoryCNF
	{
	public:
		//function should return false when no more input is available
		virtual bool Finished() = 0;

		//Interface method to read the CNF stream. 
		//The first two numbers returned should be N (number of variables) and M (number of clauses)
		//Then for each clause, the participating literals, with appropriate sign,
		//terminated by zero. Just like reading the numbers from a CNF file one by one.
		virtual int NextInt() = 0;

		virtual ~MemoryCNF();
	};
}

#endif

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


#ifndef ANALOGSAT_CNFREADER
#define ANALOGSAT_CNFREADER

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../problem/SatProblem.h"

namespace analogsat
{
	//lightweight cnf file reader
	// based on https://github.com/vegard/cnf-utils/blob/master/cnf-sort-clauses.cc
	class CnfReader
	{
	private:
		FILE *f;		
		char message[1024];

	public:

		CnfReader(); //read from stdin
		CnfReader(const char* filename); //read from file
		~CnfReader();

		//reads the file contents into a SatProblem. If fails, then an empty satproblem is returned, and GetMessage() tells what went wrong.
		//Optional function pointer: the function is called when the IO part is over and the SatProblem is being constructed.
		bool Read(SatProblem& problem, void(*pingFunc)() = NULL);

		const char* GetMessage() const;
		
	};
}

#endif

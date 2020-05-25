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


#ifndef ANALOGSAT_CNFWRITER
#define ANALOGSAT_CNFWRITER

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../problem/SatProblem.h"

namespace analogsat
{
	class CnfWriter
	{
	private:

		FILE *f;
		char message[1024];

	public:

		CnfWriter(); //write to stdout
		CnfWriter(const char* filename); //write to file
		~CnfWriter();

		//writes the given problem to file. returns true if successful. if not, GetMessage() tells what went wrong.
		bool Write(const SatProblem& problem, void(*pingFunc)() = NULL);
		const char* GetMessage() const;

	};
}

#endif
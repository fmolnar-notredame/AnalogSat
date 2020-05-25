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


#ifndef ANALOGSAT_ISATMAKER
#define ANALOGSAT_ISATMAKER

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "SatProblem.h"

namespace analogsat
{
	//abstract factory to SAT problems
	class ISatMaker
	{
	public:
		virtual ~ISatMaker() {}

		virtual void MakeSatProblem(SatProblem& problem) = 0;
	};
}

#endif
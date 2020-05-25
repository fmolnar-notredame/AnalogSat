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


#include "SatResult.h"

namespace analogsat
{

	const char* GetSatResultMessage(SatResult result)
	{
		switch (result)
		{
		case SatResult::SAT_SOLUTION_FOUND: return "solution found";
		case SatResult::SAT_UNDERFLOW: return "underflow";		
		case SatResult::SAT_ODE_INTERRUPTED: return "user interrupt";
		case SatResult::SAT_MAXTIME_REACHED: return "maximum time reached";
		case SatResult::SAT_MAXITER_REACHED: return "maximum steps reached";
		default: return "unknown result";
		}
	}

}
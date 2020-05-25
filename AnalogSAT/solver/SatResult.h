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


#ifndef ANALOGSAT_RESULT_H
#define ANALOGSAT_RESULT_H

namespace analogsat
{
	//result of solving a SAT problem
	enum SatResult
	{
		SAT_SOLUTION_FOUND,		//a solution was found
		SAT_UNDERFLOW,			//ODE integration encountered an underflow (step size went to zero), cannot continue		
		SAT_ODE_INTERRUPTED,	//callback requested stop
		SAT_MAXTIME_REACHED,	//maxtime reached without finding a solution
		SAT_MAXITER_REACHED,	//maximum number of iterations reached without finding a solution
		SAT_UNKNOWN				//??
	};

	//get a short string describing the SAT result
	const char* GetSatResultMessage(SatResult result);
}

#endif

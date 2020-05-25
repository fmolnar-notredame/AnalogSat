//This file is part of Minisat for AnalogSAT
//Copyright(C) 2019  Ferenc Molnar
//
//Minisat for AnalogSAT is free software : you can redistribute it and / or modify
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

#ifndef MINISAT_MAIN_H
#define MINISAT_MAIN_H

// main include file to compile minisat into another project, specifically to include it with AnalogSAT

#include <vector>
#include <functional>
#include "../utils/MemParse.h"

namespace Minisat
{
	struct SolverStats
	{
		unsigned long long restarts, conflicts, decisions, rnd_decisions, propagations, tot_literals, max_literals;
		double cpu_time;

		//print stats just like minisat would
		void Print();
	};

	struct Results
	{
		int exitcode;		//return value from the main run function
		char message[1024]; //printed error message, in case of errors
		char answer[128];	//final answer (SATISFIABLE or UNSATISFIABLE)
		SolverStats stats;	//solver statistics
		std::vector<int> solution; //signed variable indices

		Results();
		Results(const Results& other);
		Results& operator=(const Results& other);
	};		

	// Run Minisat on the given CNF file (must be text, compressed files are NOT supported).
	// Optional arguments can be specified, as if running from the command line (argc should reflect only the args passed to this function)
	// Notably, arguments related to writing the solution to a file are still supported.
	// An std::functional may be passed, which is called every 100k conflicts (about 1 sec on a high-end cpu in 2019). 
	// If the function returns false, solving gets interrupted.
	Results SolveWithMinisat(const char* input_file, int argc = 0, char** argv = NULL, std::function<bool()> ping = nullptr);

	// Run Minisat using in-memory CNF provider.
	// Optional arguments can be specified, as if running from the command line (argc should reflect only the args passed to this function).
	// Notably, arguments related to writing the solution to a file are still supported.
	// An std::functional may be passed, which is called every 100k conflicts (about 1 sec on a high-end cpu in 2019). 
	// If the function returns false, solving gets interrupted.
	Results SolveWithMinisat(MemoryCNF& parser, int argc = 0, char** argv = NULL, std::function<bool()> ping = nullptr);

}

#endif
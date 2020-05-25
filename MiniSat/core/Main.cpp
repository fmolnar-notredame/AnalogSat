/*
* My Extension to Minisat for compatibility with AnalogSAT
* Copyright 2019 Ferenc Molnar
*
* Original Copyright and License information below.
*/

/*****************************************************************************************[Main.cc]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

// ---------------------------------------------------------------------------------------------
// Modified by Ferenc Molnar (2019) 
// Modifications allow Minisat to be compiled as a static library, cross-platform, either Linux or Windows.
// Also added a callback function: if provided, this function is called from Minisat every time when
// progress would be reported (regardless of verbosity level). This function must return true, otherwise solving is interrupted.
// To use, include "minisat.h" and link with the library.
//----------------------------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS 1 // visual studio...

// headers to main minisat functions
#include <errno.h>
#include <signal.h>

#include "../utils/System.h"
#include "../utils/ParseUtils.h"
#include "../utils/Options.h"
#include "../core/Dimacs.h"
#include "../core/Solver.h"
#include <vector>

#include "../include/minisat.h"

namespace Minisat
{

	// fwd-declare
	Results run(int argc, char** argv, std::function<bool()> ping);
	Results run2(MemoryCNF& parser, int argc, char** argv, std::function<bool()> ping);

	// implement empty
	MemoryCNF::~MemoryCNF(){};

	//create a new SolverStats object from the given solver
	SolverStats MakeStats(const Solver& solver, double cpu_time)
	{
		SolverStats stats;
		stats.restarts = solver.starts;
		stats.conflicts = solver.conflicts;
		stats.decisions = solver.decisions;
		stats.rnd_decisions = solver.rnd_decisions;
		stats.propagations = solver.propagations;
		stats.tot_literals = solver.tot_literals;
		stats.max_literals = solver.max_literals;
		stats.cpu_time = cpu_time;
		return stats;
	}

	//print stats just like minisat would
	void SolverStats::Print()
	{
		printf("restarts              : %" PRIu64 "\n", restarts);
		printf("conflicts             : %-12" PRIu64 "   (%.0f /sec)\n", conflicts, conflicts / cpu_time);
		printf("decisions             : %-12" PRIu64 "   (%4.2f %% random) (%.0f /sec)\n", decisions, (float)rnd_decisions * 100 / (float)decisions, decisions / cpu_time);
		printf("propagations          : %-12" PRIu64 "   (%.0f /sec)\n", propagations, propagations / cpu_time);
		printf("conflict literals     : %-12" PRIu64 "   (%4.2f %% deleted)\n", tot_literals, (max_literals - tot_literals) * 100 / (double)max_literals);
	}

	//default ctor for results
	Results::Results()
	{
		exitcode = 0;
		sprintf(message, "");
		sprintf(answer, "");
	}

	Results::Results(const Results& other)
	{
		memcpy(message, other.message, sizeof(message));
		memcpy(answer, other.answer, sizeof(answer));
		stats = other.stats;
		exitcode = other.exitcode;
		solution = other.solution;
	}

	Results& Results::operator=(const Results& other)
	{
		memcpy(message, other.message, sizeof(message));
		memcpy(answer, other.answer, sizeof(answer));
		stats = other.stats;
		exitcode = other.exitcode;
		solution = other.solution;
		return *this;
	}

	// Run Minisat on the given CNF file. Optional arguments can be specified, as if running from the command line.
	Results SolveWithMinisat(const char* input_file, int argc, char** argv, std::function<bool()> ping)
	{
		//mimic actual argc and argv
		char** arg = new char*[argc + 2];

		arg[0] = new char[2048];
		strcpy(arg[0], "minisat");

		arg[1] = new char[2048];
		strcpy(arg[1], input_file);

		//add the remaining args
		for (int i = 0; i < argc; i++)
		{
			arg[i + 2] = new char[strlen(argv[i]) + 2];
			strcpy(arg[i + 2], argv[i]);
		}

		Results r = run(argc + 2, arg, ping);

		//free
		for (int i = 0; i < argc + 2; i++)
			delete[] arg[i];
		delete[] arg;

		return r;
	}

	// Run Minisat using in-memory CNF. Optional arguments can be specified, as if running from the command line.
	Results SolveWithMinisat(MemoryCNF& parser, int argc, char** argv, std::function<bool()> ping)
	{
		//mimic actual argc and argv
		char** arg = new char*[argc + 1];

		arg[0] = new char[2048];
		strcpy(arg[0], "minisat");

		//add the remaining args
		for (int i = 0; i < argc; i++)
		{
			arg[i + 1] = new char[strlen(argv[i]) + 1];
			strcpy(arg[i + 1], argv[i]);
		}

		Results r = run2(parser, argc + 1, arg, ping);

		//free
		for (int i = 0; i < argc + 1; i++)
			delete[] arg[i];
		delete[] arg;

		return r;
	}

	//main part of solving, assuming the solver has been configured and ready to go
	Results run_solver(Solver& S, const char* outfile)
	{
		Results r;

		// record start time
		double start_time = cpuTime();

		// open outout file if specified
		FILE* res = (outfile != nullptr) ? fopen(outfile, "wb") : NULL;

		// unit propagation
		if (!S.simplify())
		{
			if (res != NULL) fprintf(res, "UNSAT\n"), fclose(res);
			sprintf(r.message, "Solved by unit propagation");
			r.stats = MakeStats(S, cpuTime() - start_time);
			sprintf(r.answer, "UNSATISFIABLE");
			r.exitcode = 20;
			return r;
		}

		// solving
		vec<Lit> dummy;
		lbool ret = S.solveLimited(dummy);

		// collect stats
		r.stats = MakeStats(S, cpuTime() - start_time);
		sprintf(r.answer, ret == l_True ? "SATISFIABLE" : ret == l_False ? "UNSATISFIABLE" : "INDETERMINATE");

		// report result on stdout
		printf("%s\n", r.answer);

		// write results to file, if one was specified
		if (res != NULL)
		{
			if (ret == l_True)
			{
				fprintf(res, "SAT\n");
				for (int i = 0; i < S.nVars(); i++)
					if (S.model[i] != l_Undef)
						fprintf(res, "%s%s%d", (i == 0) ? "" : " ", (S.model[i] == l_True) ? "" : "-", i + 1);
				fprintf(res, " 0\n");
			}
			else if (ret == l_False)
				fprintf(res, "UNSAT\n");
			else
				fprintf(res, "INDET\n");
			fclose(res);
		}

		// copy results in memory
		if (ret == l_True)
		{
			for (int i = 0; i < S.nVars(); i++)
			{
				if (S.model[i] != l_Undef) //skip undefined variables
				{
					r.solution.push_back((S.model[i] == l_True) ? (i + 1) : -(i + 1));
				}
			}
		}

		return r;

	}

	// original Minisat code, simplified, linux-only parts removed, interrupt-handlers removed
	Results run(int argc, char** argv, std::function<bool()> ping)
	{
		parseOptions(argc, argv, true);

		Solver S;
		S.verbosity = 1;
		if (ping != nullptr) S.setPingFunction(ping);

		FILE *in = fopen(argv[1], "r");
		if (in == NULL)
		{
			Results r;
			sprintf(r.message, "ERROR! Could not open file: %s\n", argv[1]);
			r.exitcode = 1;
			return r;
		}

		parse_DIMACS(in, S);
		fclose(in);

		return run_solver(S, (argc >= 3) ? argv[2] : nullptr);
	}

	Results run2(MemoryCNF& parser, int argc, char** argv, std::function<bool()> ping)
	{
		parseOptions(argc, argv, true);

		Solver S;
		S.verbosity = 1;
		if (ping != nullptr) S.setPingFunction(ping);

		parse_Memory_DIMACS(parser, S);

		return run_solver(S, (argc >= 3) ? argv[2] : nullptr);
	}
}

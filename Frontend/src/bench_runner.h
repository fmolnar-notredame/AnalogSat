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


#ifndef FRONTEND_RUN_BENCH_H
#define FRONTEND_RUN_BENCH_H

#include <random>
#include <vector>

#define ANALOGSAT_INCLUDE_UTILS
#include "analogsat.h"
#include "minisat.h"
#include "config.h"


//main controller function to run benchmark series
// powers: N (problem size) is computed as 2^power * 10, allows for geometric series of problem sizes
void RunBenchSeries(Configuration conf, std::vector<double> powers);

// Measure the CPU/GPU speedup.
// For a given configuration, do:
// Cycle through the problem index and run existing problems with some short walltime limit.
// Add up the total elapsed time across samples.
// Repeat until the total elapsed computational walltime reaches the walltimeout limit.
// powers: N (problem size) is computed as 2^power * 10, allows for geometric series of problem sizes
void RunSpeedtest(Configuration conf, std::vector<double> powers, double walltimeout);

class BenchRunner
{
private:

	Configuration conf;

	//size of the current problem
	int N, M;	//variables, clauses
	
	analogsat::FastSatMaker rnd;
	analogsat::SatProblem problem;
	bool cnf_exists;	//indicates that the current SAT problem has been written to a cnf file.

	//std::string problem_folder;
	//std::string result_folder;

	char problemName[1024];	//file name with relative path
	char infile[1024];	//input CNF file name	
	char outfile[1024]; //results go here for a given problem
	std::vector<std::string> doneSamples; // problemNames that have been done already

	analogsat::CpuRandom<double>* rand_cpu;
	analogsat::CudaRandom<double>* rand_gpu; //make on demand

	//AnalogSolverFamily solverfamily; //which type of CTDS to use

	//double timeout; //seconds, walltime. Solving is aborted if this time is exceeded.
	double lastSolveTime; //seconds, walltime of last Run

	template <typename TFloat, typename TState>
	void Configure_Solver(analogsat::ISat<TFloat, TState>* ctds, analogsat::SatSolver<TFloat, TState>* solver);

	//run the current problem with minisat, save results to predetermined file, return true if the problem is solved
	bool RunMinisat();

	//run the current problem with analogsat on cpu, save results to predetermined file, return true if the problem is solved
	bool RunAnalogsatCpu();	

	//run the current problem with analogsat on gpu, save results to predetermined file, return true if the problem is solved
	bool RunAnalogsatGpu();

public:
	BenchRunner(Configuration _conf);
	~BenchRunner();


	//set the wallclock timeout for running the SAT solvers
	//void SetTimeout(double seconds);

	// get ready for a new problem
	void Configure(int _N, int sampleID);

	// run SAT solver, save problem if it is new and found to be SAT
	void RunSample();

	// get the walltime (seconds) of the last run
	double GetLastSolveTime() const;

	//return true if the currently configured problem has been loaded from disk (if not, then a random problem will be made)
	bool CnfExists() const;

	//return true if the currently configured SAT problem has already been solved by any solver
	bool IsDone() const;

};



#endif

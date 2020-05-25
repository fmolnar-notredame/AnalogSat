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


#ifndef ANALOGSAT_RUN_RAMSEY_H
#define ANALOGSAT_RUN_RAMSEY_H

#define ANALOGSAT_INCLUDE_UTILS
#include "analogsat.h"
#include "minisat.h"
#include "config.h"

//main controller functino to run Ramsey problems
void RunRamseySeries(Configuration conf);

class RamseyRunner
{
private:

	Configuration conf;

	int K, N, M; // sat problem parameters
	int NN; // number of nodes in the graph
	double alpha;	

	analogsat::SatProblem problem;


	char problemName[1024];	//file name with relative path
	char infile[1024];	//input CNF file name	
	char outfile[1024]; //results go here for a given problem

	analogsat::CpuRandom<double>* rand_cpu;
	analogsat::CudaRandom<double>* rand_gpu;

	double lastSolveTime; //seconds, walltime of last Run
	std::vector<double> solution; //solution state vector

	bool RunMinisat();
	bool RunAnalogsatCpu();
	bool RunAnalogsatGpu();

	template <typename TFloat, typename TState>
	void Configure_Solver(analogsat::ISat<TFloat, TState>* ctds, analogsat::SatSolver<TFloat, TState>* solver);

public:
	RamseyRunner(Configuration _conf);
	~RamseyRunner();

	// get ready for a new configuration of problem
	void Configure(std::vector<int> R, int _NN);

	// get the walltime (seconds) of the last run
	double GetLastSolveTime() const;

	//get the last solution vector
	const std::vector<double>& GetSolution() const;

	// solve the configured problem
	bool RunSample();

};



#endif

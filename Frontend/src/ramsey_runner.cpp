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


#define _CRT_SECURE_NO_WARNINGS 1 // pacify VS

#include "ramsey_runner.h"
#include "minisat_adapter.h"
#include "make_ramsey_problems.h"

#include <sstream>
#include <string>

using namespace analogsat;
using namespace Minisat;
using namespace std;


void RunRamseySeries(Configuration conf)
{
	//parse the Ramsey string
	vector<int> R;
	stringstream ss(conf.ramsey);
	string token;
	while (std::getline(ss, token, ','))
	{
		R.push_back(atoi(token.c_str()));
	}
	
	conf.EnsureProblemFolder();
	conf.EnsureResultFolder();

	RamseyRunner rr(conf);
	char matrixname[1024];	
	
	for (int N = (int)conf.nStart; N <= (int)conf.nEnd; N++)
	{
		rr.Configure(R, N);
		RamseyClauseMaker rcm(N, R, conf.ramseycircular ? RAMSEY_CIRCULAR : RAMSEY_NONE, N);
		sprintf(matrixname, "%s/coloring_%s_N%d.dat", conf.resultFolder, GetRamseyDigits(R).c_str(), N);

		for (int s = conf.sampleStart; s < conf.sampleEnd; s++)
		{
			if (rr.RunSample())
			{
				rcm.WriteColorMatrix(matrixname, rr.GetSolution());
			}
		}
	}	
	
}

// configure solver, ctds, and rk parameters
template <typename TFloat, typename TState>
void RamseyRunner::Configure_Solver(ISat<TFloat, TState>* ctds, SatSolver<TFloat, TState>* solver)
{
	ctds->Set_B(conf.bias);

	solver->SetProblem(problem);
	solver->SetMaxTime(conf.tmax);
	solver->SetMaxSteps(conf.stepmax);
	solver->SetBatchSize(conf.batch);
}

//run the given problem with minisat, save results to predetermined file
//return true if the problem is solved, OR indeterminate (return false only if UNSAT for sure)
bool RamseyRunner::RunMinisat()
{
	WallClock clock;
	clock.Start();

	// building internal data structures is part of runtime (file parsing is not)
	MinisatAdapter adapter(problem);

	// run minisat with a wallclock limit
	Minisat::Results results = Minisat::SolveWithMinisat(adapter, 0, 0, [&]()
	{
		return clock.GetTotalElapsedTime() < conf.timeout; //seconds
	});

	clock.Stop();
	bool solved = strcmp(results.answer, "SATISFIABLE") == 0;

	lastSolveTime = clock.GetTotalElapsedTime();
	
	//make solution vector (analogsat compatible) manually
	solution.resize(problem.Get_N());
	for (int item : results.solution)
	{
		if (item > 0)
		{
			solution[item - 1] = 1.0;
		}
		else
		{
			solution[-item - 1] = -1.0;
		}
	}

	// Output Format: sampleName, CPUID, solver, answer, walltime, cputime, rhscount (only for analogsat)
	string cpuid = GetCpuName();
	FILE *f = fopen(outfile, "a");
	fprintf(f, "%s\t%s\tminisat\t%s\t%lf\t%lf\t0\n",
		problemName,
		cpuid.c_str(),
		results.answer,
		clock.GetTotalElapsedTime(),
		results.stats.cpu_time);
	fclose(f);

	return solved;
}


bool RamseyRunner::RunAnalogsatCpu()
{
	//report
	printf("Running on CPU\n");
	printf("  %d-SAT\n", problem.Get_K());
	printf("  clauses: %d\n", problem.Get_M());
	printf("  variables: %d\n\n", problem.Get_N());

	//time integrator
	CpuRungeKutta<double>* rk = new CpuRungeKutta<double>();
	rk->SetEpsilon(conf.eps);

	//CTDS
	ISat<double, CpuODEState<double>>* ctds;
	char solver_suffix[10];
	switch (conf.family)
	{
	case ANALOGSAT_ORIGINAL:
		ctds = new CpuSat<double>();
		strcpy(solver_suffix, "orig");
		break;
	case ANALOGSAT_TANH:
		ctds = new CpuSatTanh<double>();
		strcpy(solver_suffix, "tanh");
		break;
	default:
		throw runtime_error("unknorn solver family");
	}

	//Solver
	CpuSatSolver<double> solver(ctds, rk);
	
	//Configure
	Configure_Solver(ctds, &solver);

	//Initial condition
	if (rand_cpu == NULL) rand_cpu = new CpuRandom<double>();
	solver.SetRandomInitialState(*rand_cpu);

	int minViolation = problem.Get_M();
	vector<double> state;
	int n = problem.Get_N();
	int m = problem.Get_M();

	solver.SetCallback([&](CpuSatSolver<double>::Interface& solvr)->bool
	{
		int violation = solvr.GetLastClauseViolationCount();
		if (violation < minViolation) minViolation = violation;

		double wall = solvr.GetElapsedWallTime();
		int steps = solvr.GetStepCount();		

		if (steps % 100 == 0)
		{
			printf("  %d\t%5.4lf\t%e\t%d\t%d\n", steps, solvr.GetElapsedTime(), solvr.GetLastStepSize(), violation, minViolation);
		}

		if (wall > conf.timeout)
		{
			printf("  walltime limit reached\n");
			return false;
		}
		else return true;
	});

	SatResult res = solver.Solve();
	bool solved = (res == SAT_SOLUTION_FOUND);
	solver.GetSatState().CopyTo(solution);

	printf("%s\n", GetSatResultMessage(res));

	//save stat
	// Output Format: sampleName, CPUID, solver, answer, walltime, cputime, rhscount
	string cpuid = GetCpuName();
	FILE *f = fopen(outfile, "a");
	fprintf(f, "%s\t%s\tanalogsat0%s\t%s\t%lf\t%lf\t%d\n",
		problemName,
		cpuid.c_str(),
		solver_suffix,
		GetSatResultMessage(res),
		solver.GetElapsedWallTime(),
		solver.GetElapsedCpuTime(),
		solver.GetRhsCount());
	fclose(f);

	lastSolveTime = solver.GetElapsedWallTime();

	return solved;
}


bool RamseyRunner::RunAnalogsatGpu()
{
	//report
	printf("Running on GPU\n");
	printf("  %d-SAT\n", problem.Get_K());
	printf("  clauses: %d\n", problem.Get_M());
	printf("  variables: %d\n\n", problem.Get_N());

	//CTDS
	ISat<double, CudaODEState<double>>* ctds;
	char solver_suffix[10];
	if (conf.family == ANALOGSAT_ORIGINAL)
	{
		strcpy(solver_suffix, "orig");
		switch (conf.solverVersion)
		{
		case 1: ctds = new CudaSat1<double>(); break;
		case 2: ctds = new CudaSat2<double>(); break;
		case 3: ctds = new CudaSat3<double>(); break;
		default: throw runtime_error("invalid cuda solver"); break;
		}
	}
	else if (conf.family == ANALOGSAT_TANH)
	{
		strcpy(solver_suffix, "tanh");
		switch (conf.solverVersion)
		{
		case 1: ctds = new CudaSatTanh1<double>(); break;
		case 2: ctds = new CudaSatTanh2<double>(); break;
		case 3: ctds = new CudaSatTanh3<double>(); break;
		default: throw runtime_error("invalid cuda solver"); break;
		}
	}
	else runtime_error("unknorn solver family");

	//time integrator
	CudaRungeKutta<double>* rk = new CudaRungeKutta<double>();	
	rk->SetEpsilon(conf.eps);

	//Solver
	CudaSatSolver<double> solver(ctds, rk);

	//Configure
	Configure_Solver(ctds, &solver);
	
	//Initial condition
	if (rand_gpu == NULL) rand_gpu = new CudaRandom<double>();
	solver.SetRandomInitialState(*rand_gpu);

	int minViolation = problem.Get_M();
	vector<double> state;
	int n = problem.Get_N();
	int m = problem.Get_M();

	solver.SetCallback([&](CudaSatSolver<double>::Interface& solvr)->bool
	{
		int violation = solvr.GetLastClauseViolationCount();
		if (violation < minViolation) minViolation = violation;

		double wall = solvr.GetElapsedWallTime();
		int steps = solvr.GetStepCount();

		if (steps % 100 == 0)
		{
			printf("  %d\t%5.4lf\t%e\t%d\t%d\n", steps, solvr.GetElapsedTime(), solvr.GetLastStepSize(), violation, minViolation);
		}

		if (wall > conf.timeout)
		{
			printf("  walltime limit reached\n");
			return false;
		}
		else return true;
	});

	SatResult res = solver.Solve();
	bool solved = (res == SAT_SOLUTION_FOUND);
	solver.GetSatState().CopyTo(solution);

	printf("%s\n", GetSatResultMessage(res));

	//save stat
	// Output Format: sampleName, GPUID, solver, answer, walltime, cputime, rhscount
	string gpuid = GetGpuName();
	FILE *f = fopen(outfile, "a");
	fprintf(f, "%s\t%s\tanalogsat%d%s\t%s\t%lf\t%lf\t%d\n",
		problemName,
		gpuid.c_str(),
		conf.solverVersion,
		solver_suffix,
		GetSatResultMessage(res),
		solver.GetElapsedWallTime(),
		solver.GetElapsedCpuTime(),
		solver.GetRhsCount());
	fclose(f);

	lastSolveTime = solver.GetElapsedWallTime();

	return solved;
}


RamseyRunner::RamseyRunner(Configuration _conf)
	: conf(_conf)
{
	conf.EnsureResultFolder();
	conf.EnsureProblemFolder();

	//defaults	
	rand_cpu = NULL; //make them on demand (so CPU only code can run this without GPU)
	rand_gpu = NULL;
}

RamseyRunner::~RamseyRunner()
{
	NULLDEL(rand_cpu);
	NULLDEL(rand_cpu);
}

// get ready for a new Ramsey problem
void RamseyRunner::Configure(vector<int> R, int _NN)
{
	//get problem and file names based on specifications
	sprintf(problemName, GetRamseyFileName(R, _NN, conf.ramseycircular).c_str());
	sprintf(infile, "%s/%s", conf.problemFolder, problemName);
	sprintf(outfile, "%s/perf_%s_N%d.dat", conf.resultFolder, GetRamseyDigits(R).c_str(), _NN);

	//load
	if (FileExists(infile))
	{		
		CnfReader r(infile);
		r.Read(problem);
		NN = _NN;
	}
	else throw runtime_error(string("ramsey input cnf file '") + string(infile) + string("' does not exist"));

	K = problem.Get_K();
	N = problem.Get_N();
	M = problem.Get_M();
	alpha = (double)M / (double)N;
}

bool RamseyRunner::RunSample() 
{
	printf("Running Ramsey CNF with K=%d, N=%d, M=%d\n", K, N, M);

	switch (conf.type)
	{
	case MINISAT:
		printf("MINISAT:\n");
		return RunMinisat();		
		break;

	case ANALOGSAT_CPU:
		printf("ANALOGSAT CPU:\n");
		return RunAnalogsatCpu();
		break;

	case ANALOGSAT_GPUv1:
	case ANALOGSAT_GPUv2:
	case ANALOGSAT_GPUv3:
		printf("ANALOGSAT GPUv%d:\n", conf.solverVersion);
		return RunAnalogsatGpu();
		break;
	default:
		return false; //uh oh?
	}
}

const std::vector<double>& RamseyRunner::GetSolution() const
{
	return solution;
}
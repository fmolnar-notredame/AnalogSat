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


// run the minisat / analogsat benchmarks

// benchmarking:
// make a random SAT
// pass it thru the indicated solver, with wallclock timeout
// write out results:
// UUID, solver, walltime, cputime, is_solved
// outfile identifies N and M
// problem itself is kept (written as cnf) if it was found SAT for sure

#define _CRT_SECURE_NO_WARNINGS 1 // pacify VS

#include "bench_runner.h"
#include "minisat_adapter.h"

using namespace analogsat;
using namespace Minisat;
using namespace std;

//convert exponents to problem size
vector<int> GetNValues(const vector<double>& powers)
{
	vector<int> nValues;
	for (double p : powers)
	{
		double pp = pow(2.0, p);
		int N = (int)round(10.0 * pp);
		nValues.push_back(N);
	}
	return nValues;
}

// run benchmark problems
void RunBenchSeries(Configuration conf, vector<double> powers)
{
	//convert exponents to N values (problem sizes)
	vector<int> nValues = GetNValues(powers);

	// runner
	BenchRunner bench(conf);	

	//cycle through N values one by one
	for (int n : nValues)
	{		
		for (int s = conf.sampleStart; s < conf.sampleEnd; s++)
		{
			bench.Configure(n, s);

			if (conf.rerun || !bench.IsDone())
			{
				bench.RunSample();
			}
		}
	}
}

// Measure the CPU/GPU speedup
// For a given configuration, do:
// Cycle through the problem index and run existing problems with some short walltime limit.
// Add up the total elapsed time across samples.
// Repeat until the total elapsed computational walltime reaches the walltimeout limit.
void RunSpeedtest(Configuration conf, vector<double> powers, double walltimeout)
{
	//convert exponents to N values (problem sizes)
	vector<int> nValues = GetNValues(powers);

	//runner
	BenchRunner bench(conf);

	//cycle through N values one by one
	for (int n : nValues)
	{
		int s = 0; //sample index
		double totalElapsed = 0.0;
		int count = 0;
		while (totalElapsed < walltimeout && count < 200) //very small or very short problems: do only 200 at most
		{
			//load sample
			bench.Configure(n, s);
			if (bench.CnfExists())
			{
				//run it until timeout (most likely not solved, it will report elapsed time and RHS count)
				bench.RunSample();
				totalElapsed += bench.GetLastSolveTime();
				count++;
				s++;
			}
			else if (s == 0)
			{
				throw runtime_error("no cnf samples exist with the given configuration");
			}
			else
			{
				//start over the loop of samples
				s = 0;
			}
		}
	}
}



//hyperparameters for the CTDS
template <typename TFloat, typename TState>
void BenchRunner::Configure_Solver(ISat<TFloat, TState>* ctds, SatSolver<TFloat, TState>* solver)
{	
	//dynamics
	ctds->Set_B(conf.bias);

	//solver
	solver->SetProblem(problem);
	solver->SetMaxTime(conf.tmax);
	solver->SetMaxSteps(conf.stepmax);
	solver->SetBatchSize(conf.batch);
}


//run the given problem with minisat, save results to predetermined file
//return true if the problem is determined to be SAT, return false otherwise
bool BenchRunner::RunMinisat()
{
	WallClock clock;
	clock.Start();

	// building internal data structures is part of walltime! (file parsing is not)
	MinisatAdapter adapter(problem);

	// run minisat with a wallclock limit
	Minisat::Results results = Minisat::SolveWithMinisat(adapter, 0, 0, [&]()
	{
		return clock.GetTotalElapsedTime() < conf.timeout; //seconds
	});

	clock.Stop();
	bool solved = strcmp(results.answer, "SATISFIABLE") == 0;

	lastSolveTime = clock.GetTotalElapsedTime();

	if (solved)
	{
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
	}

	return solved;
}

bool BenchRunner::RunAnalogsatCpu()
{
	//time integrator
	CpuRungeKutta<double>* rk = new CpuRungeKutta<double>();
	rk->SetEpsilon(conf.eps); //1e-6 usually

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

	//function that is called every <batch> iterations
	solver.SetCallback([&](CpuSatSolver<double>::Interface& solvr)->bool
	{
		int violation = solvr.GetLastClauseViolationCount();
		if (violation < minViolation) minViolation = violation;

		double wall = solvr.GetElapsedWallTime();
		int steps = solvr.GetStepCount();
		
		if (steps % 1000 == 0)
		{
			printf("  %d\t%5.4lf\t%e\t%d\t%d\n", steps, solvr.GetElapsedTime(), solvr.GetLastStepSize(), violation, minViolation);
		}

		if (wall > conf.timeout)
		{
			printf("  walltime limit reached\n");
			return false;
		}

		return true;
	});

	SatResult res = solver.Solve();
	bool solved = (res == SAT_SOLUTION_FOUND);
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

bool BenchRunner::RunAnalogsatGpu()
{
	//time integrator
	CudaRungeKutta<double>* rk = new CudaRungeKutta<double>();
	rk->SetEpsilon(conf.eps); //usually 1e-6

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


BenchRunner::BenchRunner(Configuration _conf)
	: conf(_conf)
{
	//create the folder for results		
	//if (!DirectoryExists(conf.resultFolder)) CreateDir(conf.resultFolder);
	conf.EnsureResultFolder();

	//create the folder for problems
	//if (!DirectoryExists(conf.problemFolder)) CreateDir(conf.problemFolder);
	conf.EnsureProblemFolder();

	//init ptr
	rand_cpu = NULL;
	rand_gpu = NULL;
}

BenchRunner::~BenchRunner()
{
	NULLDEL(rand_cpu);
	NULLDEL(rand_gpu);	
}

double BenchRunner::GetLastSolveTime() const
{
	return lastSolveTime;
}


// get ready for a new problem, make one if it does not exist
void BenchRunner::Configure(int _N, int sampleID)
{
	//if the config did not change (only possibly the sample), do not bother reloading some stuff
	bool sameConfig = (N == _N);
	N = _N;	
	M = (int)(conf.alpha * N);
	int a = (int)(conf.alpha * 100); //for file naming

	//does this random problem category exist?
	char path[1024];
	sprintf(path, "%s/%da%d", conf.problemFolder, conf.k, a);
	if (!DirectoryExists(path)) CreateDir(path);

	//does this N exist for this problem category?
	char path2[1024];
	sprintf(path2, "%s/N%d", path, N);
	if (!DirectoryExists(path2)) CreateDir(path2);

	//does the SAT problem sample exist?
	sprintf(infile, "%s/p%04d.cnf", path2, sampleID);
	sprintf(problemName, "%da%d/N%d/p%04d.cnf", conf.k, a, N, sampleID);
	printf("Problem: %s\n", problemName);

	if (FileExists(infile))
	{
		//load the existing problem
		CnfReader r(infile);
		r.Read(problem);
		cnf_exists = true; 
	}
	else
	{
		cnf_exists = false;
	}

	//formulate the output filename for benchmark results
	sprintf(outfile, "%s/perf_K%d_a%d_N%d.dat", conf.resultFolder, conf.k, a, N);

	if (sameConfig) return;

	//find out which samples have been solved
	doneSamples.clear();

	if (FileExists(outfile))
	{
		FILE *f = fopen(outfile, "r");
		char line[1024];
		while (!feof(f))
		{
			//read line
			char* read = fgets(line, sizeof(line), f);
			if (read == NULL) break;

			//find first tab
			char* pch = (char*)memchr(line, '\t', strlen(line));
			if (pch == NULL) continue;

			//truncate the string right there
			*pch = 0;

			//convert to std::string and add to list
			doneSamples.push_back(string(line));
		}
		fclose(f);
	}
}


void BenchRunner::RunSample()
{
	bool result;

	if (!cnf_exists)
	{
		//make new problem
		rnd.Set_K(conf.k);
		rnd.Set_N(N);
		rnd.Set_M(M);
		printf("Making K=%d, N=%d, M=%d\n", conf.k, N, M);
		rnd.MakeSatProblem(problem);
	}

	switch (conf.type)
	{
	case MINISAT:
		printf("MINISAT:\n");
		result = RunMinisat();
		break;

	case ANALOGSAT_CPU:
		printf("ANALOGSAT CPU:\n");
		result = RunAnalogsatCpu();
		break;

	case ANALOGSAT_GPUv1:
	case ANALOGSAT_GPUv2:
	case ANALOGSAT_GPUv3:
		printf("ANALOGSAT GPUv%d:\n", conf.solverVersion);
		result = RunAnalogsatGpu();
		break;
	}

	// new problem found to be SAT? write it out (also do it if forced)
	if ((result && !cnf_exists) || conf.force_save_cnf)
	{
		//write CNF out
		printf("saving cnf\n");
		cnf_exists = true;
		CnfWriter w(infile);
		w.Write(problem);
	}

	if (result) doneSamples.push_back(problemName);
}

bool BenchRunner::CnfExists() const
{
	return cnf_exists;
}

bool BenchRunner::IsDone() const
{
	for (const auto& item : doneSamples)
	{
		if (strcmp(problemName, item.c_str()) == 0) return true;
	}
	return false;
}
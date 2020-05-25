#pragma warning (disable : 4996)  //pacify VS, "deprecated posix name"

#include <vector>
#include "cnf_runner.h"

#define ANALOGSAT_INCLUDE_UTILS
#include "analogsat.h"

#include "minisat_adapter.h"

using namespace std;
using namespace analogsat;

//write a given state to a CNFRunner file (only for double-precision)
template <class T>
void WriteState(FILE *f, int N, vector<double>& state, T& solvr)
{
	solvr.GetSatState().CopyTo(state);
	fprintf(f, "%f", solvr.GetElapsedTime());
	for (int i = 0; i < N; i++)
	{
		fprintf(f, " %f", state[i]);
	}
	fprintf(f, "\n");
}


SatResult SolveOnGpu(const SatProblem& problem, const Configuration& conf, vector<double>* lastState = NULL)
{
	//make CTDS
	ISat<double, CudaODEState<double>>* ctds;
	char familySuffix[10];
	if (conf.family == ANALOGSAT_ORIGINAL)
	{
		strcpy(familySuffix, "orig");
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
		strcpy(familySuffix, "tanh");
		switch (conf.solverVersion)
		{
		case 1: ctds = new CudaSatTanh1<double>(); break;
		case 2: ctds = new CudaSatTanh2<double>(); break;
		case 3: ctds = new CudaSatTanh3<double>(); break;
		default: throw runtime_error("invalid cuda solver"); break;
		}
	}
	else runtime_error("unknorn solver family");

	ctds->Set_B(conf.bias);

	//make integrator
	CudaRungeKutta<double>* rk = new CudaRungeKutta<double>();
	rk->SetEpsilon(conf.eps);

	//Solver setup
	CudaSatSolver<double> solver(ctds, rk);
	solver.SetProblem(problem);
	solver.SetMaxTime(conf.tmax);  //should be more than enough
	solver.SetMaxSteps(conf.stepmax);
	solver.SetBatchSize(conf.batch);

	//init state
	int minViolation = problem.Get_M();
	int n = problem.Get_N();
	int m = problem.Get_M();
	vector<double> state(n + m);

	//Random initial condition
	CudaRandom<double> rand;
	solver.SetRandomInitialState(rand);

	//CNFRunner file
	FILE *f = NULL;
	if (conf.trajectory)
	{
		char fname[1024];
		string problemName = GetFileNameWithoutExtension(GetFileNameWithoutPath(conf.cnf_file));
		sprintf(fname, "%s/%s_traj_%s.dat", conf.resultFolder, problemName.c_str(), familySuffix);
		f = fopen(fname, "w");
		if (f == NULL) throw runtime_error("could not open CNFRunner output file");

		//record at t=0, both spins and aux variables
		WriteState(f, n + m, state, solver);
	}


	//callback
	solver.SetCallback([&](CudaSatSolver<double>::Interface& solvr)->bool
	{
		int violation = solvr.GetLastClauseViolationCount();
		if (violation < minViolation) minViolation = violation;

		double wall = solvr.GetElapsedWallTime();
		int steps = solvr.GetStepCount();

		printf("  %d\t%5.4lf\t%e\t%d\t%d\n", steps, solvr.GetElapsedTime(), solvr.GetLastStepSize(), violation, minViolation);

		//save CNFRunner
		if (conf.trajectory) WriteState(f, n + m, state, solvr);

		return true;
	});

	//make it go
	SatResult result = solver.Solve();

	//close CNFRunner output
	if (conf.trajectory) fclose(f);

	//get the last state (contains the solution if found)
	if (lastState != NULL)
		solver.GetSatState().CopyTo(*lastState);

	//report basic performance
	double elapsed = solver.GetElapsedWallTime();
	printf("Performance: %f RK steps/sec\n", (double)solver.GetRhsCount() / 6 / elapsed);
	return result;

}

SatResult SolveOnCpu(const SatProblem& problem, const Configuration& conf, vector<double>* lastState = NULL)
{
	//make CTDS
	ISat<double, CpuODEState<double>>* ctds;
	char familySuffix[10];
	switch (conf.family)
	{
	case ANALOGSAT_ORIGINAL:
		ctds = new CpuSat<double>();
		sprintf(familySuffix, "orig");
		break;
	case ANALOGSAT_TANH:
		ctds = new CpuSatTanh<double>();
		sprintf(familySuffix, "tanh");
		break;
	default:
		throw runtime_error("invalid solver family");
	}

	ctds->Set_B(conf.bias);

	//make integrator
	CpuRungeKutta<double>* rk = new CpuRungeKutta<double>();
	rk->SetEpsilon(conf.eps);

	//Solver setup
	CpuSatSolver<double> solver(ctds, rk);
	solver.SetProblem(problem);
	solver.SetMaxTime(conf.tmax);  //should be more than enough
	solver.SetMaxSteps(conf.stepmax);
	solver.SetBatchSize(conf.batch);	

	//init state
	int minViolation = problem.Get_M();	
	int n = problem.Get_N();
	int m = problem.Get_M();
	vector<double> state(n + m);

	//Random initial condition
	CpuRandom<double> rand;
	solver.SetRandomInitialState(rand);

	//CNFRunner file
	FILE *f = NULL;
	if (conf.trajectory)
	{
		char fname[1024];
		string problemName = GetFileNameWithoutExtension(GetFileNameWithoutPath(conf.cnf_file));		
		sprintf(fname, "%s/%s_traj_%s.dat", conf.resultFolder, problemName.c_str(), familySuffix);
		f = fopen(fname, "w");
		if (f == NULL) throw runtime_error("could not open CNFRunner output file");

		//record at t=0, both spins and aux variables
		WriteState(f, n + m, state, solver);
	}


	//callback
	solver.SetCallback([&](CpuSatSolver<double>::Interface& solvr)->bool
	{
		int violation = solvr.GetLastClauseViolationCount();
		if (violation < minViolation) minViolation = violation;

		double wall = solvr.GetElapsedWallTime();
		int steps = solvr.GetStepCount();

		printf("  %d\t%5.4lf\t%e\t%d\t%d\n", steps, solvr.GetElapsedTime(), solvr.GetLastStepSize(), violation, minViolation);

		//save CNFRunner
		if (conf.trajectory) WriteState(f, n + m, state, solvr);

		return true;
	});

	//make it go
	SatResult result = solver.Solve();
	
	//close CNFRunner output
	if (conf.trajectory) fclose(f);

	//get the last state (contains the solution if found)
	if (lastState != NULL)
		solver.GetSatState().CopyTo(*lastState);

	//report basic performance
	double elapsed = solver.GetElapsedWallTime();
	printf("Performance: %f RK steps/sec\n", (double)solver.GetRhsCount() / 6 / elapsed);
	return result;
}

SatResult SolveUsingMinisat(const SatProblem& problem, const Configuration& conf, vector<double>* lastState = NULL)
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
	SatResult result = strcmp(results.answer, "SATISFIABLE") == 0 ? SatResult::SAT_SOLUTION_FOUND : SatResult::SAT_UNKNOWN;

	//copy solution if any
	if (lastState != NULL)
	{
		lastState->resize(problem.Get_N());
		for (size_t i = 0; i < problem.Get_N(); i++)
			(*lastState)[i] = results.solution[i] ? 1.0 : -1.0;
	}

	return result;
}

void RunCnf(Configuration conf)
{
	//folder
	conf.EnsureResultFolder();	

	//read input
	SatProblem problem;
	{
		CnfReader r(conf.cnf_file);
		if (!r.Read(problem)) throw runtime_error(r.GetMessage());
	}

	SatResult result = SatResult::SAT_UNKNOWN;
	vector<double> lastState;

	switch (conf.type)
	{
	case SolverType::ANALOGSAT_CPU:
		result = SolveOnCpu(problem, conf, &lastState);
		break;
	case SolverType::ANALOGSAT_GPUv1:
	case SolverType::ANALOGSAT_GPUv2:
	case SolverType::ANALOGSAT_GPUv3:
		result = SolveOnGpu(problem, conf, &lastState);
		break;
	case SolverType::MINISAT:
		result = SolveUsingMinisat(problem, conf, &lastState);
		break;
	default:
		throw runtime_error("unknown solver type");
	}

	printf("%s\n", GetSatResultMessage(result));

	//solver has finished, write results
	string problemName = GetFileNameWithoutExtension(GetFileNameWithoutPath(conf.cnf_file));
	char fname[1024];
	sprintf(fname, "%s/%s.out", conf.resultFolder, problemName.c_str());
	FILE *f = fopen(fname, "w");
	if (f == 0) throw runtime_error("cannot open result file to write");

	//result file: output compatible with minisat
	switch (result)
	{
	case SatResult::SAT_SOLUTION_FOUND:
		fprintf(f, "SAT\n");
		break;
	default: //AnalogSAT is an incomplete solver. It cannot determine if a problem is UNSAT.
		fprintf(f, "INDET\n");
		break;
	}

	if (problem.Get_N() > 0) fprintf(f, "%d", lastState[0] > 0 ? 1 : -1);
	for (int i = 1; i < problem.Get_N(); i++)
	{
		fprintf(f, " %d", lastState[i] > 0.0 ? i + 1 : -i - 1);
	}
	fprintf(f, " 0\n"); //terminate solution with zero, like minisat does
	fclose(f);

}
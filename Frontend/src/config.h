#ifndef FRONTEND_CONFIG_H
#define FRONTEND_CONFIG_H

#include "solver_type.h"

//configuration of the analog SAT solver frontend
//not all fields are used by all methods
class Configuration
{	
public:
	char problemFolder[1024];	//random problems are stored here when created
	char resultFolder[1024];	//results of running problems are stored here (regardless of the path in cnf_file)
	char cnf_file[1024];		//problem file name (with full path)

	AnalogSolverFamily family;  //original or Tanh CTDS (calculated from tanh)
	SolverType type;			//cpu or which gpu (calculated from nogpu and solverVersion)

	bool nogpu;			//disable any CUDA (compute on CPU)
	int cudadevice;		//used in CudaSetDevice()
	int solverVersion;	//for gpu: v1, v2, or v3
	bool minisat;		//use minisat (disable any CUDA, ignore related parameters, run on CPU)

	double eps;		//adaptive time step relative error tolerance
	double tmax;	//analog time limit
	double bias;	//bias term for highly symmetric SAT problems

	double timeout; //wallclock time limit
	int stepmax;	//max number of steps in time-integration
	int batch;		//number of steps between checks for SAT solution

	double nStart;	//problem size start value (N = 2^nStart * 10 for random, used directly for Ramsey) 
	double nEnd;	//problem size end value (N = 2^nEnd * 10 for random, used directly for Ramsey)
	double nStep;	//problem size step value

	double alpha;		//clause to variable ratio, for random problem benchmarks
	int k;				//number of literals per clause, for random problems	
	int sampleStart;	//start index of samples (inclusive)
	int sampleEnd;		//end index of samples (exclusive)

	bool trajectory;		//save trajectory
	bool use_tanh;			//use tanh CTDS instead of original
	bool rerun;			    //true: run all samples in the given range. false: run only unsolved problems.
	bool force_save_cnf;	//save the random problem as cnf even if it was not solved (or found to be UNSAT)

	char ramsey[1024];		//Ramsey string to identify Ramsey graph coloring problems
	bool ramseycircular;	//if true, Ramsey problems are made with a cirular adjacency matrix coloring constraint

	void EnsureSolverType();	//set the type (SolverType) based on the configured settings
	void EnsureSolverFamily();	//set the family (AnalogSolverFamily) based on the configured settings
	
	void EnsureProblemFolder();	//make sure the problem folder exists
	void EnsureResultFolder();	//make sure the result folder exists

private:
	void TrimPath(char* path); //remove the trailing slash or backslash from a path

};


#endif

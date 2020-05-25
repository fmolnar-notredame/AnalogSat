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


#define _SCL_SECURE_NO_WARNINGS 1  //pacify VS
#define _CRT_SECURE_NO_WARNINGS 1

#include "cnf_runner.h"
#include "bench_runner.h"
#include "make_ramsey_problems.h"
#include "ramsey_runner.h"
#include <climits>

using namespace analogsat;
using namespace std;

//common size for "strings"
#define STRLEN 1024


// parsing helper functions
template<typename T>
T StringToValue(const char* str) { return (T)0; }

template<>
int StringToValue<int>(const char* str) { return atoi(str); }

template<>
double StringToValue<double>(const char* str) { return atof(str); }

//attempt to parse the given option from arguments. returns true if found and successful, false if not found, exits if input is bad
//index i is the current index in the args
template<typename T>
bool ParseOption(string option, const vector<string>& args, T& target, size_t& i, const T min, const T max)
{
	if (i >= args.size())
	{
		fprintf(stderr, "low level parse error\n");
		exit(-1);
	}

	if (args[i] != option) return false;

	if (i >= args.size() - 1) //require: at least 2 more arguments available
	{
		string msg("insufficient arguments for '");
		msg += option;
		msg += "'\n";
		fprintf(stderr, msg.c_str());
		exit(-1);
		return false;
	}

	i++;
	T value = StringToValue<T>(args[i].c_str());
	if (value < min || value > max)
	{
		printf("value out of range: %s %s\n", args[i - 1].c_str(), args[i].c_str());
		exit(-1);
		return false;
	}
	target = value;
	i++;
	return true;
}

//special case: string argument (limits ignored, value strcpy'ed into target, which should be the target char[0] by ref)
template<>
bool ParseOption<char>(string option, const vector<string>& args, char &target, size_t& i, const char min, const char max)
{
	if (i >= args.size())
	{
		fprintf(stderr, "low level parse error\n");
		exit(-1);
	}

	if (args[i] != option) return false; // no match

	if (i >= args.size() - 1) //require: at least 2 more arguments available
	{
		string msg("insufficient arguments for '");
		msg += option;
		msg += "'\n";
		fprintf(stderr, msg.c_str());
		exit(-1);
		return false;
	}

	i++;
	strcpy(&target, args[i].substr(0, std::min((size_t)(STRLEN-1), args[i].size())).c_str());
	i++;
	return true;

}

//special case: parameterless switch interpreted as boolean True when specified (limits ignored)
template<>
bool ParseOption<bool>(string option, const vector<string>& args, bool &target, size_t& i, const bool min, const bool max)
{
	if (args[i] == option)
	{
		target = true;
		i++;
		return true;
	}
	else return false;
}


enum FrontendCommand
{
	RUN,
	BENCH,
	SPEEDTEST,
	RAMSEY_MAKE,
	RAMSEY_RUN,
	UNKNOWN
};

// process all options given by args, starting from offset ii
void ParseOptions(const vector<string>& args, Configuration& conf, size_t ii)
{
	size_t argc = args.size();
	char configname[STRLEN];

	//parse remaining arguments
	while (ii < argc)
	{
		//special case of config from a file
		if (ParseOption<char>("-config", args, configname[0], ii, 'c', 'c'))
		{
			FILE *f = fopen(configname, "r");
			if (f == NULL) throw runtime_error("cannot open config file");

			vector<string> conf_args;
			char buf[STRLEN];
			while (!feof(f) && fgets(buf, sizeof(buf), f) > 0)
			{
				buf[strcspn(buf, "\r\n")] = 0; //cut end of line in any form possible
				string line(buf);
				size_t pos = line.find_first_of(' ');
				if (pos != string::npos)
				{
					conf_args.push_back(line.substr(0, pos));
					conf_args.push_back(line.substr(pos + 1));
				}
				else conf_args.push_back(line);
			}
			fclose(f);

			ParseOptions(conf_args, conf, 0); //recursion into the config given by the file | config files can go recursive themselves...
		}
		else if (ParseOption<char>("-problem", args, conf.cnf_file[0], ii, 'c', 'c')) continue;
		else if (ParseOption<char>("-ramsey", args, conf.ramsey[0], ii, 'c', 'c')) continue;
		else if (ParseOption<bool>("-ramseycircular", args, conf.ramseycircular, ii, false, false)) continue;
		else if (ParseOption<bool>("-nogpu", args, conf.nogpu, ii, false, false)) continue;
		else if (ParseOption<int>("-usegpu", args, conf.cudadevice, ii, 0, 99)) continue;
		else if (ParseOption<bool>("-trajectory", args, conf.trajectory, ii, false, false)) continue;
		else if (ParseOption<bool>("-force_save_cnf", args, conf.force_save_cnf, ii, false, false)) continue;
		else if (ParseOption<bool>("-tanh", args, conf.use_tanh, ii, false, false)) continue;
		else if (ParseOption<bool>("-minisat", args, conf.minisat, ii, false, false)) continue;
		else if (ParseOption<bool>("-rerun", args, conf.rerun, ii, false, false)) continue;		
		else if (ParseOption<double>("-tmax", args, conf.tmax, ii, 0, 1e80)) continue;
		else if (ParseOption<double>("-bias", args, conf.bias, ii, 0.0, 1e6)) continue;
		else if (ParseOption<double>("-timeout", args, conf.timeout, ii, 0, 1e80)) continue;
		else if (ParseOption<int>("-k", args, conf.k, ii, 2, 256)) continue;		
		else if (ParseOption<int>("-stepmax", args, conf.stepmax, ii, 0, INT_MAX)) continue;
		else if (ParseOption<int>("-batch", args, conf.batch, ii, 1, INT_MAX)) continue;
		else if (ParseOption<double>("-eps", args, conf.eps, ii, 1e-8, 1e-1)) continue;
		else if (ParseOption<double>("-alpha", args, conf.alpha, ii, 0.0, 1e10)) continue;
		else if (ParseOption<int>("-samplestart", args, conf.sampleStart, ii, 0, 9999)) continue;
		else if (ParseOption<int>("-sampleend", args, conf.sampleEnd, ii, 0, 10000)) continue;
		else if (ParseOption<double>("-nstart", args, conf.nStart, ii, 1.0, 100.0)) continue;
		else if (ParseOption<double>("-nend", args, conf.nEnd, ii, 1.0, 100.0)) continue;
		else if (ParseOption<double>("-nstep", args, conf.nStep, ii, 0.0, 100.0)) continue;
		else if (ParseOption<char>("-resultfolder", args, conf.resultFolder[0], ii, 'c', 'c')) continue;
		else if (ParseOption<char>("-problemfolder", args, conf.problemFolder[0], ii, 'c', 'c')) continue;
		else if (ParseOption<int>("-version", args, conf.solverVersion, ii, 1, 3)) continue;
		else
		{			
			printf("bad argument: %s\n", args[ii].c_str());
			throw runtime_error("options parsing error");
		}
	}

}

int main(int argc, char** argv)
{
	try
	{
		//default configuration -------------
		Configuration conf;
		
		//ctds params
		conf.eps = 1e-6;
		conf.tmax = 1e8;
		conf.bias = 0.0;

		//solver params
		conf.family = AnalogSolverFamily::ANALOGSAT_ORIGINAL;
		conf.use_tanh = false;
		conf.solverVersion = 1;
		conf.type = SolverType::ANALOGSAT_GPUv1;
		conf.nogpu = false;
		conf.cudadevice = -1;

		//benchmark control
		conf.minisat = false;
		conf.trajectory = false;
		conf.rerun = false;
		conf.force_save_cnf = false;

		//running limits
		conf.timeout = 3600;
		conf.batch = 50;
		conf.stepmax = 50000000;

		//samples and iterations
		conf.sampleStart = 0;
		conf.sampleEnd = 100;
		conf.nStart = 1.0;
		conf.nEnd = 5.0;
		conf.nStep = 0.5;
		conf.k = 3;
		conf.alpha = 4.25;
		strcpy(conf.ramsey, "");
		conf.ramseycircular = false;

		//folders		
		strcpy(conf.problemFolder, ".");
		strcpy(conf.resultFolder, ".");
		
		//-------------------------------------

		//grab all args into vector of strings
		vector<string> args(argv, argv + argc);

		//parse command line arguments
		if (argc < 2)
		{
			printf("\n");
			printf("AnalogSAT Frontend\n");
			printf("==================\n");
			printf("\n");
			printf("Copyright(C) 2019 Ferenc Molnar\n");
			printf("License: GNU GPL v3.\n");
			printf("\n");
			printf("Simulations of continuous-time dynamical systems (CTDS) that solve Boolean\n");
			printf("satisfiability (SAT) problems or minimize the number of violated clauses.\n");
			printf("\n");
			printf("Usage: analogsat <command> [-option <value>]\n");
			printf("For details, see the Documentation.\n");
			printf("\n");
			printf("Commands:\n");
			printf("---------\n");
			printf("  run           Run a given SAT problem from a CNF file\n");
			printf("  bench         Measure the solving time of a series of random SAT problems\n");
			printf("  speedtest     Measure the iteration performance on a series of problems\n");
			printf("  make_ramsey   Make CNF files representing Ramsey graph coloring problems\n");
			printf("  run_ramsey    Run Ramsey graph coloring problems made by make_ramsey\n");
			printf("\n");
			printf("Options for all commands:\n");
			printf("-------------------------\n");
			printf("-version <N>  AnalogSAT GPU solver version (1/2/3) [1]\n");
			printf("-nogpu        Use the AnalogSAT CPU solver, do not call any GPU functions\n");
			printf("-minisat      Use the MiniSat solver (CPU), do not call any GPU functions\n");
			printf("-usegpu <N>   Use CUDA device number N (via CudaSetDevice(N)) (0::99) [0]\n");
			printf("-tanh         Use the alternative CTDS formulation, which is based on the\n");
			printf("              Tanh formula for evolving the auxiliary variables.\n");
			printf("-bias <F>     Coefficient for the bias term in the CTDS (0::1e6) [0]\n");
			printf("-tmax <F>     Maximum analog time for the integration (0::1e80) [1e8]\n");
			printf("-stempax <N>  Maximum number of discrete steps to take in the integration.\n");
			printf("              (0::INT_MAX) [50000000]\n");
			printf("-timeout <F>  Walltime limit for the integration, in seconds (0::1e80) [3600]\n");
			printf("-eps <F>      Relative error tolerance parameter for the adaptive time stepping\n");
			printf("              of the ODE integrator method (1e-8::1e-1) [1e-6]\n");
			printf("-batch <N>    Number of discrete steps to take at once before the current state\n");
			printf("              is checked for a SAT solution (1::INT_MAX) [50]\n");
			printf("\n");
			printf("Options for 'run' command:\n");
			printf("--------------------------\n");
			printf("-problem <S>        Name of the input CNF file.\n");
			printf("-resultfolder <S>   Folder where the results are saved. Folder is created if\n");
			printf("                    it does not exist. [.]\n");
			printf("-trajectory         Save the trajectory of the integrated CTDS. The trajectory\n");
			printf("                    is saved at the resolution specified by -batch. \n");
			printf("\n");
			printf("Options for 'bench' command:\n");
			printf("----------------------------\n");
			printf("-problemfolder <S>  Folder where the CNF files for the random problems are\n");
			printf("                    saved. Subfolders will be created for problem classes \n");
			printf("                    automatically.\n");
			printf("-resultfolder <S>   Folder where the results are saved.\n");
			printf("-force_save_cnf     Save the CNF of the current sample to the problemfolder\n");
			printf("                    regardless of satisfiability.\n");
			printf("-samplestart <N>    Start index for problem samples, inclusive (0::9999) [0]\n");
			printf("-sampleend <N>      End index for problem samples, exclusive (0::10000) [100]\n");
			printf("-k <N>              The length of clauses made in random problems (2::256) [3]\n");
			printf("-alpha <F>          The ratio of clauses made, relative to the number of\n");
			printf("                    variables in the problem (0::1e10) [4.25]\n");
			printf("-nstart <F>         Starting value of the problem size exponent, inclusive,\n");
			printf("                    (1::100) [1.0]\n");
			printf("-nend <F>           Ending value of the problem size exponent, inclusive,\n");
			printf("                    (1::100) [5.0]\n");
			printf("-nstep <F>          Step value of the problem size exponent (0::100) [0.5]\n");
			printf("-rerun              Run the given sample, even if it has been solved before\n");
			printf("\n");
			printf("Options for 'speedtest' command:\n");
			printf("--------------------------------\n");
			printf("-problemfolder <S>  Folder from where the CNF files are loaded.\n");
			printf("-resultfolder <S>   Folder where the results are saved.\n");
			printf("-samplestart <N>    Start index for problem samples, inclusive (0::9999) [0]\n");
			printf("-sampleend <N>      End index for problem samples, exclusive (0::10000) [100]\n");
			printf("-alpha <F>          The ratio of clauses made, relative to the number of\n");
			printf("                    variables in the problem (0::1e10) [4.25]\n");
			printf("-nstart <F>         Starting value of the problem size exponent, inclusive,\n");
			printf("                    (1::100) [1.0]\n");
			printf("-nend <F>           Ending value of the problem size exponent, inclusive,\n");
			printf("                    (1::100) [5.0]\n");
			printf("-nstep <F>          Step value of the problem size exponent (0::100) [0.5]\n");
			printf("\n");
			printf("Options for 'make_ramsey' command:\n");
			printf("----------------------------------\n");
			printf("-R <S>              Comma-separated list of integers that specify the\n");
			printf("                    Ramsey problem. E.g., -R 3,4,5 for problem R(3,4,5).\n");
			printf("-problemfolder <S>  Folder where the CNF files for Ramsey problems are saved.\n");
			printf("-nstart <N>         Starting size N for the number of nodes in the graph.\n");
			printf("-nend <N>           Ending size N for the number of nodes in the graph \n");
			printf("                    (inclusive).\n");
			printf("-ramseycircular     Add the constraint that the adjacency matrix of the graph\n");
			printf("                    is a circulant matrix. \n");
			printf("\n");
			printf("Options for 'run_ramsey' command:\n");
			printf("---------------------------------\n");
			printf("-R <S>              Comma-separated list of integers that specify the\n");
			printf("                    Ramsey problem. E.g., -R 3,4,5 for problem R(3,4,5).\n");
			printf("-problemfolder <S>  Folder where the CNF files for Ramsey problems are saved.\n");
			printf("-resultfolder <S>   Folder where results are saved.\n");
			printf("-nstart <N>         Starting size N for the number of nodes in the graph.\n");
			printf("-nend <N>           Ending size N for the number of nodes in the graph \n");
			printf("                    (inclusive).\n");
			printf("-ramseycircular     Add the constraint that the adjacency matrix of the graph\n");
			printf("                    is a circulant matrix. \n");
			printf("\n");
			return 0;
		}
		
		//start parsing, first get the command
		FrontendCommand command = FrontendCommand::UNKNOWN;		
		size_t ii = 1;
		bool dummy;

		if (ParseOption<bool>("run", args, dummy, ii, false, false)) command = FrontendCommand::RUN;
		else if (ParseOption<bool>("bench", args, dummy, ii, false, false)) command = FrontendCommand::BENCH;
		else if (ParseOption<bool>("speedtest", args, dummy, ii, false, false)) command = FrontendCommand::SPEEDTEST;
		else if (ParseOption<bool>("make_ramsey", args, dummy, ii, false, false)) command = FrontendCommand::RAMSEY_MAKE;
		else if (ParseOption<bool>("run_ramsey", args, dummy, ii, false, false)) command = FrontendCommand::RAMSEY_RUN;
		else
		{
			fprintf(stderr, "Error: '%s'\n", args[ii].c_str());
			throw runtime_error("analogsat command not understood");
		}

		//parse all remaining args (recursive)
		ParseOptions(args, conf, ii);

		//select cuda device when needed (initializes CUDA, so don't call under -nogpu)
		if (!conf.nogpu && !conf.minisat && conf.cudadevice >= 0)
		{
			cudaSetDevice(conf.cudadevice);
		}

		//select the right solver
		conf.EnsureSolverType();
		conf.EnsureSolverFamily();


		// ---- Execute ---------------------------------------------------------------------

		switch (command)
		{
		case FrontendCommand::RUN:
			RunCnf(conf);
			break;

		case FrontendCommand::BENCH: //benchmark on random problems
			{
				vector<double> powers;
				for (double n = conf.nStart; n < conf.nEnd + 1e-3; n += conf.nStep) 
					powers.push_back(n); //end+1e-3 against roundoff errors

				RunBenchSeries(conf, powers);
			}
			break;

		case FrontendCommand::SPEEDTEST:
			{
				vector<double> powers;
				for (double n = conf.nStart; n < conf.nEnd + 1e-3; n += conf.nStep)
					powers.push_back(n); //end+1e-3 against roundoff errors

				RunSpeedtest(conf, powers, 5.0); //5 second total running walltime
			}
			break;

		case FrontendCommand::RAMSEY_MAKE:
			MakeRamseyProblems(conf);
			break;

		case FrontendCommand::RAMSEY_RUN:
			RunRamseySeries(conf);
			break;

		default:
			printf("Not yet implemented\n");
			break;
		}
	
		printf("All done.\n");
	}
	catch (exception &ex)
	{
		printf("Error: %s\n", ex.what());
		return -1;
	}

	return 0;
}
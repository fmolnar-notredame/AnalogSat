#include <vector>
#include "config.h"

#define ANALOGSAT_INCLUDE_UTILS
#include "analogsat.h"

using namespace std;
using namespace analogsat;

void Configuration::EnsureSolverType()
{
	if (minisat)
	{
		type = SolverType::MINISAT;
	}
	else if (nogpu)
	{
		type = SolverType::ANALOGSAT_CPU;
	}
	else
	{
		switch (solverVersion)
		{
		case 1:
			type = SolverType::ANALOGSAT_GPUv1; break;
		case 2:
			type = SolverType::ANALOGSAT_GPUv2; break;
		case 3:
			type = SolverType::ANALOGSAT_GPUv3; break;
		}
	}
}

void Configuration::EnsureSolverFamily()
{
	if (use_tanh) family = AnalogSolverFamily::ANALOGSAT_TANH;
	else family = AnalogSolverFamily::ANALOGSAT_ORIGINAL;
}

void Configuration::TrimPath(char* path)
{
	int i = (int)strlen(path) - 1;
	while (i >= 0 && (path[i] == '/' || path[i] == '\\'))
	{
		path[i] = 0;
		i--;
	}
}

void Configuration::EnsureProblemFolder()
{
	//ensure the output folder tree exists
	vector<string> tree = GetPathSuccessive(problemFolder);
	for (string dir : tree)
	{
		//printf("%s\n", dir.c_str());
		if (!DirectoryExists(dir))
		{
			//printf("MKDIR %s\n", dir.c_str());
			CreateDir(dir);
		}
	}

	//trim trailing slash symbols
	TrimPath(problemFolder);
}

void Configuration::EnsureResultFolder()
{
	//ensure the output folder tree exists
	vector<string> tree = GetPathSuccessive(resultFolder);
	for (string dir : tree)
	{
		//printf("%s\n", dir.c_str());
		if (!DirectoryExists(dir))
		{
			//printf("MKDIR %s\n", dir.c_str());
			CreateDir(dir);
		}
	}

	//trim trailing slash symbols
	TrimPath(resultFolder);
}



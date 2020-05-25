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


#define _CRT_SECURE_NO_WARNINGS 1 //pacify VS

#include "make_ramsey_problems.h"
#include <sstream>
#include <string>

using namespace std;
using namespace analogsat;

std::string GetRamseyDigits(const std::vector<int>& R)
{
	if (R.size() > 8) throw runtime_error("invalid ramsey digits");
	
	char digit[20];
	std::string result = "";
	for (int r : R)
	{
		sprintf(digit, "%d", r);
		result += digit;
	}

	return result;
}

std::string GetRamseyFileName(const std::vector<int>& R, int N, bool circular)
{
	//compose filename
	char fname[1024];
	char digit[20];
	if (circular) sprintf(fname, "ramsey_circular_");
	else sprintf(fname, "ramsey_regular_");

	for (int r : R)
	{
		sprintf(digit, "%d", r);
		strcat(fname, digit);
	}

	sprintf(digit, "_N%d.cnf", N);
	strcat(fname, digit);

	return std::string(fname);
}

// make a CNF file for a Ramsey problem
void MakeRamseyCNF(const vector<int>& R, int N, bool circular, const string& folder)
{
	//make the ramsey CNF clauses
	RamseyClauseMaker rcm(N, R, circular ? RAMSEY_CIRCULAR : RAMSEY_NONE, N);

	//get the name
	std::string fname = GetRamseyFileName(R, N, circular);
	std::string fullname = folder + "/" + fname;
	
	//write it
	printf("Saving %s\n", fname.c_str());
	rcm.WriteCNF(fullname.c_str(), true);
}


//Make CNF for a given range of Ramsey problems.
//R(5, 5) problems:
//43 <= R(5,5) <= 48, so there is no known coloring for N=43. 
//But if you find one, you will be famous.
//R(3, 3, 3, 3) problems:
//51 <= R(3,3,3,3) <= 62, so there is no known coloring for N=51. 
//But if you find one, you will be famous.
void MakeRamseyProblems(Configuration conf)
{
	//parse the Ramsey string
	vector<int> R;
	stringstream ss(conf.ramsey);
	string token;
	while (std::getline(ss, token, ','))
	{
		R.push_back(atoi(token.c_str()));
	}

	//place to put cnf files
	conf.EnsureProblemFolder();
	string folder(conf.problemFolder);

	//make it go
	for (int N = (int)conf.nStart; N <= (int)conf.nEnd; N++)
	{
		MakeRamseyCNF(R, N, conf.ramseycircular, folder);
	}
}
//This file is part of AnalogSAT
//Copyright(C) 2019 Ferenc Molnar
//
//AnalogSAT is free software: you can redistribute it and / or modify
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


#define _CRT_SECURE_NO_WARNINGS 1

#include <stdexcept>
#include "CnfReader.h"

using namespace std;

namespace analogsat
{
	CnfReader::CnfReader()
	{
		f = stdin;
		strcpy(message, "");
	}

	CnfReader::CnfReader(const char* filename)
	{
		f = fopen(filename, "r");
		strcpy(message, "");
	}

	CnfReader::~CnfReader()
	{
		if (f != NULL && f != stdin)
		{
			fclose(f);
			f = NULL;
		}
	}

	bool CnfReader::Read(SatProblem& problem, void(*pingFunc)())
	{
		if (f == NULL)
		{
			sprintf(message, "file open failed");
			return false;
		}

		//allocate vector for clause lengths		
		char line[1024];
		int n=-1, m=-1;

		//read preamble 
		while (!feof(f))
		{
			char c = fgetc(f);

			if (c == 'p')
			{
				//format declaration
				if (fscanf(f, " cnf %u %u\n", &n, &m) == 2) break;
				else
				{
					sprintf(message, "expected: p cnf <variables> <clauses>");
					return false;
				}
			}
			else
			{
				//read line and ignore				
				fgets(line, sizeof(line), f);
			}
		}

		if (n < 0 || m < 0)
		{
			sprintf(message, "could not find a SAT declaration");
			return false;
		}
		else if (n == 0 || m == 0) //added support for empty SAT problems, which is OK!
		{
			sprintf(message, "empty problem");
			problem.Clear();
			return true;
		}

		//allocate the length values for each clause
		std::vector<int> lengths;
		lengths.reserve(m);

		//allocate for the literals - assume minimum 3-sat, can still grow if it must
		std::vector<int> literals;
		literals.reserve(m * 3);

		int clausesRead = 0;
		while (clausesRead < m && !feof(f))
		{
			//read next clause
			int length = 0;
			while (true)
			{
				int literal;
				if (fscanf(f, "%d", &literal) != 1)
				{
					sprintf(message, "expected more literals in file");
					return false;
				}

				if (literal == 0) break; //literal completed

				literals.push_back(literal);
				length++;
			}

			//record clause length
			lengths.push_back(length);
			clausesRead++;
		}

		if (clausesRead < m)
		{
			sprintf(message, "file is missing %d clauses", m - clausesRead);
			return false;
		}

		sprintf(message, "read ok");

		if (pingFunc != NULL) pingFunc();

		//compose the satproblem
		problem.Clear();
		problem.AddClauses(literals, lengths);
		return true;
	}

	const char* CnfReader::GetMessage() const { return message; }
}

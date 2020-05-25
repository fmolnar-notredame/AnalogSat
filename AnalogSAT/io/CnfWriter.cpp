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
#include "CnfWriter.h"
#include <algorithm>

using namespace std;

namespace analogsat
{
	CnfWriter::CnfWriter()
	{
		f = stdout;
		strcpy(message, "");
	}

	CnfWriter::CnfWriter(const char* filename)
	{
		f = fopen(filename, "w");
		strcpy(message, "");
	}

	CnfWriter::~CnfWriter()
	{
		if (f != NULL && f != stdout)
		{
			fclose(f);
			f = NULL;
		}
	}

	bool CnfWriter::Write(const SatProblem& problem, void(*pingFunc)())
	{
		if (f == NULL)
		{
			sprintf(message, "file open failed");
			return false;
		}

		int N = problem.Get_N();
		int M = problem.Get_M();

		if (pingFunc != NULL) pingFunc();

		fprintf(f, "p cnf %d %d\n", N, M);
		for (int j = 0; j < M; j++)
		{
			const auto& clause = problem.GetClause(j);
			for (int i : clause)
			{
				fprintf(f, "%d ", i);
			}
			fprintf(f, "0\n");
		}

		return true;
	}

	const char* CnfWriter::GetMessage() const { return message; }
}
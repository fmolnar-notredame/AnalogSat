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


#ifndef RAMSEY_PROBLEM_MAKER_H
#define RAMSEY_PROBLEM_MAKER_H

#include "ramsey_maker.h"
#include "config.h"


//get the number sequence for Ramsey problems
std::string GetRamseyDigits(const std::vector<int>& R);

//get a standardized file name for Ramsey problems
std::string GetRamseyFileName(const std::vector<int>& R, int N, bool circular);

// make a CNF file for a Ramsey problem
void MakeRamseyCNF(const std::vector<int>& R, int N, bool circular, const std::string& folder);

//Top level method to make CNF for a given range of Ramsey problems.
void MakeRamseyProblems(Configuration conf);


#endif

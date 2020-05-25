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


#ifndef ANALOGSAT_SOLVERTYPE_H
#define ANALOGSAT_SOLVERTYPE_H

//selection of SAT solver
enum SolverType
{
	MINISAT,
	ANALOGSAT_CPU,
	ANALOGSAT_GPUv1,
	ANALOGSAT_GPUv2,
	ANALOGSAT_GPUv3,
};

enum AnalogSolverFamily
{
	ANALOGSAT_ORIGINAL,
	ANALOGSAT_TANH
};

#endif

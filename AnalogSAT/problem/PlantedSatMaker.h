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


#ifndef ANALOGSAT_PLANTEDSATMAKER_H
#define ANALOGSAT_PLANTEDSATMAKER_H

#include "RandomSatMaker.h"

namespace analogsat
{
	//planted SAT problem generator - makes a random SAT problem with a planted solution
	//SAT solution is guaranteed
	//clauses are not repeated and they do not have the same variable twice
	class PlantedSatMaker : public RandomSatMaker
	{
	public:
		PlantedSatMaker();
		PlantedSatMaker(unsigned long seed);
		~PlantedSatMaker() override;

		//make the configured SAT problem
		void MakeSatProblem(SatProblem& problem) override;

		//return the bool solution planted in the last created SatProblem
		std::vector<bool> GetPlantedSolution();
		
		//return the planted solution as a public state vector (including aux, set to zeros)
		template<typename TFloat>
		std::vector<TFloat> GetPlantedSolutionAsFloat()
		{
			std::vector<TFloat> sol(n + m, (TFloat)0);
			for (int i = 0; i < n; i++) sol[i] = planted[i] ? (TFloat)1.0 : (TFloat)-1.0;
			return sol;
		}

	protected:

		std::vector<bool> planted;
		bool Satisfied(); // checks if the current clause is satisfied by the planted state
	};
}

#endif

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


#ifndef RAMSEY_CNF_MAKER_H
#define RAMSEY_CNF_MAKER_H

#include "analogsat.h"

//types of coloring constraints of complete graphs
enum RamseyGraphConstraint
{
	RAMSEY_NONE,		//no special constraints
	RAMSEY_CIRCULAR		//circular constraints, use constraintArg to specify circular block size
};


//Make CNF clauses from any Ramsey problem
//The Ramsey problem is specified by the "Ramsey digits",
//e.g., Ramsey problem R(5, 5) is given by vector<int> R = {5, 5};
//The methods are not particularly optimized, because they are fast enough already.
class RamseyClauseMaker
{
public:
	//Make Ramsey problems on N nodes
	//If a coloring constraint is given then blocksize specifies the size of the block in the adjacency matrix where it is applied.
	//Typically, blocksize should be N itself. Blocksize should always be a divisor of N.
	RamseyClauseMaker(int _N, std::vector<int> R, RamseyGraphConstraint _constraint, int _blocksize = 0);

	~RamseyClauseMaker();

	//get the result
	const std::vector<std::vector<int>>& GetClauses();

	//set one edge (given by adjacency matrix row and column) to have a specific color
	//implemented as adding unit clauses
	void SetSpecificColor(int row, int column, int color);

	//set the indicated edge color as an initial condition in a state vector, which is compatible with the analogsat::SatProblem made from this object's clauses
	void SetInitialCondition(int row, int column, int color, std::vector<double>& ic);

	//write the adjacency matrix with colors specified by the given solution state vector, and write it to a file
	//returns the color matrix
	std::vector<std::vector<int>> WriteColorMatrix(const char* filename, const std::vector<double>& solution);

	//convert a state vector from a sat problem with circular constraints to a sat problem with no constraints
	void ApplyCircularToRegular(const std::vector<double>& instate, std::vector<double>& outstate);

	//write the clauses to a CNF file	
	//optionally, report details on the created SAT problem
	void WriteCNF(const char* filename, bool verbose=false);

private:

	int colorDepth;			//how many bits are needed to represent a color

	std::vector<int> counter;	//recursive call counter
	std::vector<int> edges;		//buffer of edge IDs in a given clique
	std::vector<std::vector<int>> clauses; //results go here
	std::vector<int> unitClause;	//temporary buffer

	int k;		//clique size
	int N;		//graph size
	int* E;		//adjacency matrix (edge IDs)
	int edgeCount;	// count of edge IDs made
	int len;

	int* E_regular;
	int* E_circular;
	int edgeCount_regular;
	int edgeCount_circular;
	int blocksize;

	int clauseCount;

	//add clauses to exclude monochromatic cliques
	void ExcludeMonochromaticClique(int cliqueSize, int color);

	//add clauses to exclude a specific color from ever being used
	void ExcludeColor(int color);

	//recursive loop over clique edges, emit clique constraints
	void Inner(int color, int depth);

	//helper method to save an adjacency matrix to file
	void SaveMatrix(int *E);

	//create an adjacency matrix with no coloring constraints
	void MakeRegularAdjacency(int** _E, int& _edgeCount);

	//create an adjacency matrix with circular coloring constraints
	void MakeCircularAdjacency(int** _E, int& _edgeCount, int blocksize);

};

#endif
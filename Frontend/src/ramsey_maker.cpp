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


#define _CRT_SECURE_NO_WARNINGS 1  //pacify VS

#include "ramsey_maker.h"

using namespace std;
using namespace analogsat;


RamseyClauseMaker::RamseyClauseMaker(int _N, std::vector<int> R, RamseyGraphConstraint _constraint, int _blocksize)
{
	N = _N;
	blocksize = _blocksize == 0 ? N : _blocksize;

	if (R.size() < 2 || R.size() > 8) throw invalid_argument("The number of Ramsey digits must be between 2 and 8");

	int s = (int)R.size();
	int smax = 0;	//max color index (exclusive)

	switch (s)
	{
	case 2: colorDepth = 1; smax = 2; break;	//2 colors: 1-bit
	case 3:
	case 4: colorDepth = 2; smax = 4; break;	//3 or 4 colors: 2-bit
	default: colorDepth = 3; smax = 8; break;	//up to 8: 3-bit
	}

	//create edges, both types
	//we need both in order to support state vector conversions
	MakeRegularAdjacency(&E_regular, edgeCount_regular);
	MakeCircularAdjacency(&E_circular, edgeCount_circular, blocksize);

	//SaveMatrix(E_circular);

	//select which one is active
	switch (_constraint)
	{
	case RAMSEY_NONE: E = E_regular; edgeCount = edgeCount_regular; break;
	case RAMSEY_CIRCULAR: E = E_circular; edgeCount = edgeCount_circular; break;
	}

	//add exclusion constraints
	for (int i = 0; i < s; i++)
	{
		ExcludeMonochromaticClique(R[i], i);
	}

	//add exclusions for unused colors
	for (int i = s; i < smax; i++)
	{
		ExcludeColor(i);
	}
}


RamseyClauseMaker::~RamseyClauseMaker()
{
	delete[] E_regular;
	delete[] E_circular;
}

const std::vector<std::vector<int>>& RamseyClauseMaker::GetClauses() { return clauses; }

void RamseyClauseMaker::SetSpecificColor(int row, int column, int color)
{
	//edge ID of the selected edge
	int e = E[row * N + column];

	unitClause.resize(1);

	for (int i = 0; i < colorDepth; i++)
	{
		int lit = -(e * colorDepth + i + 1); //assume false color bit
		if ((color >> i) & 0x1) lit = -lit;  //switch to true color bit
		unitClause[0] = lit;
		clauses.push_back(unitClause);
	}
}

void RamseyClauseMaker::SetInitialCondition(int row, int column, int color, std::vector<double>& ic)
{
	int e = E[row * N + column];
	for (int k = 0; k < colorDepth; k++)
	{
		int var = e * colorDepth + k; // which variable (0-based index)	
		if (var >= ic.size()) ic.resize(var + 1);
		ic[var] = ((color >> k) & 0x1) ? 1.0 : -1.0; //set color bit as sat variable
	}
}

vector<vector<int>> RamseyClauseMaker::WriteColorMatrix(const char* filename, const std::vector<double>& solution)
{
	FILE *f = fopen(filename, "w");
	vector<vector<int>> matrix;
	for (int r = 0; r < N; r++)
	{
		vector<int> row(N);
		for (int c = 0; c < N; c++)
		{
			int color = 0;
			if (r == c)
			{
				color = -1;
			}
			else
			{
				int e = E[r * N + c];
				for (int k = 0; k < colorDepth; k++)
				{
					int idx = e * colorDepth + k; //solution does not have +1 variable
					if (solution[idx] > 0) color |= (0x1 << k);
				}
			}

			if (c > 0) fprintf(f, " %d", color + 1);
			else fprintf(f, "%d", color + 1);
			row[c] = color + 1;
		}
		fprintf(f, "\n");
		matrix.push_back(row);
	}
	fclose(f);
	return matrix;
}

void RamseyClauseMaker::ApplyCircularToRegular(const std::vector<double>& instate, std::vector<double>& outstate)
{
	int EE = edgeCount_regular * colorDepth;
	if (outstate.size() < EE) outstate.resize(EE); //aux vars missing, won't be able to use in SetSatState()

	//loop over regular edges, look up their edgeID in the circular matrix
	int e = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			//e is the edge ID in the outstate
			int e_source = E_circular[i * N + j]; //the edge ID in the instates

			//copy each bit from the reference edge
			for (int b = 0; b < colorDepth; b++)
			{
				int idx_source = e_source * colorDepth + b;
				int idx_target = e * colorDepth + b;
				outstate[idx_target] = instate[idx_source];
			}

			e++; //increment edges
		}
	}
}

void RamseyClauseMaker::WriteCNF(const char* filename, bool verbose)
{
	SatProblem problem;
	problem.AddClauses(clauses);

	if (verbose)
	{
		printf("  %d-SAT\n", problem.Get_K());
		printf("  clauses: %d\n", problem.Get_M());
		printf("  variables: %d\n", problem.Get_N());
		printf("  alpha: %f\n", (float)problem.Get_M() / (float)problem.Get_N());
	}

	CnfWriter w(filename);
	w.Write(problem);
}

void RamseyClauseMaker::ExcludeMonochromaticClique(int cliqueSize, int color)
{
	k = cliqueSize;

	counter.resize(k);		//allocate counters (iterators for each recursion depth)
	len = k * (k - 1) / 2;	//how many edges in the clique
	edges.resize(len);		//allocate space for edgeIDs inside a clique

	clauseCount = 0;

	//enumerate all cliques (recursive):
	//all combinations of k nodes from N
	//emit constraints for each by forbidding monochromatism for all edges within the clique
	for (int i = 0; i < N; i++)
	{
		counter[0] = i;
		Inner(color, 1);
	}

	//printf("clauses added: %d\n", clauseCount);
}

//recursive loop over clique edges, emit clique constraints
void RamseyClauseMaker::Inner(int color, int depth)
{
	if (depth == k) //full depth reached, emit edge constraint
	{
		//now we have a clique, with nodes indexed by counter[] elements
		//enumerate all edges in the clique: (k choose 2) (build clauses below for each color bit)
		int count = 0;
		for (int j1 = 0; j1 < k; j1++)
			for (int j2 = j1 + 1; j2 < k; j2++)
				edges[count++] = E[counter[j1] * N + counter[j2]];

		//check edges for non-identicality (sanity check, normally this should not happen)
		int first = edges[0];
		bool ok = false;
		for (int i = 1; i < count; i++)
		{
			if (edges[i] != first)
			{
				ok = true;
				break;
			}
		}
		if (!ok) printf("Warning: Monochromatic clique inavoidable\n");

		int clen = count * colorDepth; //number of literals = edge count * color depth
		vector<int> clause;
		clause.reserve(clen);

		//for each bit of the color, add clauses excluding that bit of monochromatic clique
		for (int i = 0; i < colorDepth; i++)
		{
			for (int j = 0; j < count; j++)
			{
				int lit = edges[j] * colorDepth + i + 1;
				if ((color >> i) & 0x1) lit = -lit; //when excluding color 0, emit a regular literal. To exclude color bit 1, emit a negated literal
				clause.push_back(lit);
			}
		}


		clauses.push_back(clause);
		clauseCount++;
	}
	else //recursion
	{
		//loop over all possible combination of nodes, counter stores node indices in the adjacency matrix
		for (int i = counter[depth - 1] + 1; i < N; i++)
		{
			counter[depth] = i;
			Inner(color, depth + 1); //recursion
		}
	}
}

void RamseyClauseMaker::ExcludeColor(int color)
{
	for (int e = 0; e < edgeCount; e++) //for all distinct edge IDs
	{
		vector<int> clause;
		clause.reserve(colorDepth);

		for (int i = 0; i < colorDepth; i++)
		{
			int lit = e * colorDepth + i + 1;
			if ((color >> i) & 0x1) lit = -lit; //when excluding color bit 0, emit a regular literal. To exclude color bit 1, emit a negated literal
			clause.push_back(lit);
		}

		clauses.push_back(clause);
	}
}

void RamseyClauseMaker::SaveMatrix(int *E)
{
	FILE *f = fopen("matrix.dat", "w");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (j > 0) fprintf(f, "  ");
			fprintf(f, "%02d", E[i*N + j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}


void RamseyClauseMaker::MakeRegularAdjacency(int** _E, int& _edgeCount)
{
	//complete graph on N nodes
	*_E = new int[N * N]; //edge IDs
	int* EE = *_E;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) EE[i * N + j] = -1; //init to none (for debug purposes)

	_edgeCount = 0; //edge index
	for (int i = 0; i < N; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			//edge ID
			EE[i * N + j] = _edgeCount;
			EE[j * N + i] = _edgeCount;

			_edgeCount++; //increment edges
		}
	}
}

void RamseyClauseMaker::MakeCircularAdjacency(int** _E, int& _edgeCount, int blocksize)
{
	if (N % blocksize != 0) throw invalid_argument("the circular block size must be a divisor of N");

	//the first row of each block are the bool variables, the rest are derived from them
	//only the upper block-triangle is required
	//variable identity is encoded into edge IDs	

	//complete graph on N nodes
	*_E = new int[N * N]; //edge IDs
	int* EE = *_E;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) EE[i * N + j] = -1;

	int blockcount = N / blocksize;
	_edgeCount = 0;

	//assign edge IDs
	for (int a = 0; a < blockcount; a++)
	{
		for (int b = a; b < blockcount; b++) //upper triangle blocks
		{
			//first col in this block is a variable IF this is not a main diagonal block!!! If it is, then skip the diagonal nodes!!!
			for (int i = (a == b ? 1 : 0); i < blocksize; i++)
			{
				int row = a * blocksize + i;
				int col = b * blocksize + 0;

				//edge ID
				EE[row * N + col] = _edgeCount;
				EE[col * N + row] = _edgeCount;	//for diagonal blocks, this sets the first row correctly
				// for offdiag blocks, this sets the blocktranspose

				_edgeCount++; //increment edges
			}

			//ShowMatrix(E, N, a*blocksize, (a + 1)*blocksize, b*blocksize, (b + 1)*blocksize);

			//for all other cols in this block, copy circularly
			for (int j = 1; j < blocksize; j++)
			{
				for (int i = 0; i < blocksize; i++)
				{
					int row = a * blocksize + i;
					int col = b * blocksize + j;

					if (EE[row * N + col] >= 0) continue; //diagonal block: this is already done

					//get the edge ID from the reference
					int srcrow = a * blocksize + ((blocksize + i - 1) % blocksize); //offset so it's not negative
					int srccol = b * blocksize + j - 1;
					int ee = EE[srcrow * N + srccol]; //row above, next column, circularly in the block

					//edge ID
					EE[row * N + col] = ee;
					EE[col * N + row] = ee;

					//ShowMatrix(E, N, a*blocksize, (a + 1)*blocksize, b*blocksize, (b + 1)*blocksize);

				}
			}
		}
	}
}

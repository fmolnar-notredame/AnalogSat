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


#ifndef ANALOGSAT_CLAUSE2WRAPHASH_H
#define ANALOGSAT_CLAUSE2WRAPHASH_H

#include "Clause.h"

namespace analogsat
{
	//provides custom hash and comparison for Clause objects
	//instance to be used in ctor of unordered_set
	class ClauseHelper
	{
	private:
		int* storage;

	public:

		ClauseHelper(int* _storage) : storage(_storage){};

		ClauseHelper() : storage(NULL) {};

		void UpdateStoragePtr(int* _storage) { storage = _storage; }

		//provide hash
		size_t operator()(const Clause& clause) const
		{
			return clause.GetHash();
		}

		//provide comparison
		bool operator() (const Clause& obj1, const Clause& obj2) const
		{
			return obj1.Equals(obj2, storage);
		}
	};
}

#endif

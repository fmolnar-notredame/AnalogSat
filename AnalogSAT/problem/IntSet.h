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


#ifndef ANALOGSAT_INTSET_H
#define ANALOGSAT_INTSET_H

#include <vector>
#include <cstdlib>

namespace analogsat
{
	//represent a set of POSITIVE integers, allow for fast lookup without hash
	class IntSet
	{
		std::vector<int> list;		//list of items in the set
		std::vector<std::size_t> lookup;	//index of the item in the list

	public:
		IntSet();

		IntSet(const IntSet& other);

		IntSet& operator=(const IntSet& other);

		void Add(int number);

		void Remove(int number);

		void Clear();

		bool Contains(int item) const;

		int Count() const;

		std::vector<int>::iterator begin();

		std::vector<int>::iterator end();
	};

}


#endif
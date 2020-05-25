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


#include "IntSet.h"

using namespace std;

namespace analogsat
{
	IntSet::IntSet()
	{
		list.reserve(100);
		list.push_back(0); //1-based offset
	}

	IntSet::IntSet(const IntSet& other) : list(other.list), lookup(other.lookup)
	{ }

	IntSet& IntSet::operator=(const IntSet& other)
	{
		list = other.list;
		lookup = other.lookup;
		return *this;
	}

	void IntSet::Add(int number)
	{
		//expand storage as needed
		if ((int)lookup.size() <= number) lookup.resize(number + 1, 0);

		if (lookup[number] == 0)
		{
			list.push_back(number);
			lookup[number] = list.size() - 1;
		} //else: already present
	}

	void IntSet::Remove(int number)
	{
		if ((int)lookup.size() <= number) return; //not present

		size_t index = lookup[number];

		//move the last stored number into its place -- ok if it's the same number, ok if it's the only stored number
		int lastNumber = list[list.size() - 1];
		list[index] = lastNumber;
		lookup[lastNumber] = index;

		//erase the old lookup entry
		lookup[number] = 0; //erased

		//shrink the list
		list.pop_back();
	}

	void IntSet::Clear()
	{
		auto end = list.end();
		for (auto it = list.begin() + 1; it != end; ++it) lookup[*it] = 0;		

		list.clear();
		list.push_back(0);
	}

	bool IntSet::Contains(int item) const
	{
		if (item >= (int)lookup.size()) return false;
		return lookup[item] > 0;
	}

	int IntSet::Count() const { return (int)list.size() - 1; }

	std::vector<int>::iterator IntSet::begin() { return list.begin() + 1; }
	std::vector<int>::iterator IntSet::end() { return list.end(); }
}


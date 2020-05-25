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


#include <algorithm>
#include "Clause.h"
#include "../util/Hash.h"
#include "../util/utils.h"

namespace analogsat
{
	void Clause::UpdateHash(const int* storage)
	{
		hashval = 0;
		if (Length > 0)
		{
			hashval ^= fnv_hash((unsigned char*)&Length, sizeof(Length));
			hashval ^= fnv_hash((unsigned char*)(storage + Start), Length * sizeof(int));
		}
	}

	//default ctor
	Clause::Clause()
	{
		Start = 0;
		Length = 0;
		hashval = 0;
	}

	//init ctor
	Clause::Clause(int _Start, int _Length)
	{
		Start = _Start;
		Length = _Length;
		hashval = 0;
	}

	//copy ctor
	Clause::Clause(const Clause& other)
	{
		Start = other.Start;
		Length = other.Length;
		hashval = other.hashval;
	}


	//equality		
	bool Clause::Equals(const Clause &other, const int* storage) const
	{
		if (other.Length != Length) return false;
		for (int i = 0; i < Length; i++) if (storage[Start + i] != storage[other.Start + i]) return false;
		return true;
	}

	//assignment
	Clause& Clause::operator=(const Clause& other)
	{
		Start = other.Start;
		Length = other.Length;
		hashval = other.hashval;
		return *this;
	}

	void Clause::Sort(std::vector<int>& storage)
	{
		//ensure sortedness
		switch (Length)
		{
		case 1: break;
		case 2: swap_if_greater(storage[Start], storage[Start + 1]); break;
		case 3: swap_if_greater(storage[Start], storage[Start + 1]); swap_if_greater(storage[Start], storage[Start + 2]); swap_if_greater(storage[Start + 1], storage[Start + 2]); break;
		default: std::sort(storage.begin() + Start, storage.begin() + Start + Length, &AbsValueComparator);
		}

	}

	std::size_t Clause::GetHash() const { return hashval; }

	
	void Clause::swap_if_greater(int& a, int& b)
	{
		if (abs(a) > abs(b)) std::swap(a, b);
	}

	int Clause::GetStart() const { return Start; }
	int Clause::GetLength() const { return Length; }

}
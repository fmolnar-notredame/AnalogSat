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


#ifndef ANALOGSAT_CLAUSE_H
#define ANALOGSAT_CLAUSE_H

#include <vector>

namespace analogsat
{
	class Clause
	{
	private:
		//data storage: 16 bytes
		std::size_t hashval;	//8
		int Start, Length;		//4+4

		void swap_if_greater(int& a, int& b);

	public:

		//default ctor
		Clause();

		//init ctor
		Clause(int _start, int _length);

		//copy ctor
		Clause(const Clause& other);

		//equality	-- custom form, use adapter class for unordered_set
		bool Equals(const Clause &other, const int* storage) const;

		//assignment
		Clause& operator=(const Clause& other);

		//hashing  -- custom form, use adapter class for unordered_set
		void UpdateHash(const int* storage);

		//sort literals in storage
		void Sort(std::vector<int>& storage);

		//getters
		int GetStart() const;
		int GetLength() const;

		//hash code
		std::size_t GetHash() const;
	};
}

#endif

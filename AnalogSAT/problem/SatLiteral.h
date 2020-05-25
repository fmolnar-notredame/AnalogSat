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


#ifndef ANALOGSAT_SATLITERAL_H
#define ANALOGSAT_SATLITERAL_H

#include <vector>

namespace analogsat
{
	//Literal iterator host object, implements enumerability for foreach loops
	class SatLiterals
	{
	public:
		typedef std::vector<int>::const_iterator iter;

	private:		
		iter startIterator;
		iter endIterator;

	public:
		SatLiterals(const std::vector<int>& storage, int start, int length) : startIterator(storage.begin() + start), endIterator(storage.begin() + start + length) {}
		SatLiterals(const SatLiterals& other) : startIterator(other.startIterator), endIterator(other.endIterator) {}

		iter begin() const { return startIterator; } 
		iter end() const { return endIterator; }
		
	};
}

#endif

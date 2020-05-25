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


#ifndef ANALOGSAT_ISATRANDOM_H
#define ANALOGSAT_ISATRANDOM_H

namespace analogsat
{
	//interface to random generators that provide for random SAT problem generation and random SAT states
	//can be implemented on either CPU or GPU
	template <class TFloat>
	class ISatRandom
	{
	public:
		virtual ~ISatRandom() { };

		//fills the given array with standard uniform randoms (values between 0 and 1)
		virtual void GenerateUniform(TFloat* addr, int length) = 0;		
	};
}

#endif

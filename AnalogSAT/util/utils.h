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


#ifndef ANALOGSAT_UTILS_H
#define ANALOGSAT_UTILS_H

#include <vector>
#include <algorithm>
#include <numeric>

#define NULLDEL(x) if (x != 0) { delete x; x = 0; }

namespace analogsat
{
	//compare values by absolute value
	bool AbsValueComparator(int a, int b);

	//compare vectors by absolute value sequentially
	bool ColComparator(const std::vector<int>& a, const std::vector<int>& b);

	//compare vectors by size
	bool VecSizeComparator(const std::vector<int>& a, const std::vector<int>& b);

	inline int iabs(int a) { return a < 0 ? -a : a; }

	//calculate the sorted indices for a vector
	template <typename T>
	std::vector<int> sort_indices(const std::vector<T>& v, bool(*comparator)(const T&, const T&))
	{
		std::vector<int> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0); //range

		std::sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {return comparator(v[i1], v[i2]); });

		return idx;
	}

	//sort a vector in-place, with added optimizations when the vector is shorter than 4 items
	void LocalSort(std::vector<int>& vec, int start, int length);
}

#endif

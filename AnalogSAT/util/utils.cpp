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


#include "utils.h"
#include <algorithm>

namespace analogsat
{

	bool AbsValueComparator(int a, int b)
	{
		return (a < 0 ? -a : a) < (b < 0 ? -b : b);
	}

	bool ColComparator(const std::vector<int>& a, const std::vector<int>& b)
	{
		int k = std::min((int)a.size(), (int)b.size());
		for (int i = 0; i < k; i++)
		{
			if (iabs(a[i]) < iabs(b[i])) return true;
			else if (iabs(a[i]) > iabs(b[i])) return false;
		}

		return false;
	}

	bool VecSizeComparator(const std::vector<int>& a, const std::vector<int>& b)
	{
		return a.size() < b.size();
	}

	void SwapIfGreater(int& a, int& b)
	{
		if (a > b) std::swap(a, b);
	}

	void LocalSort(std::vector<int>& vec, int start, int length)
	{
		switch (length)
		{
		case 0: return;
		case 1: return;
		case 2: SwapIfGreater(vec[start], vec[start + 1]); break;
		case 3: SwapIfGreater(vec[start], vec[start + 1]);
			SwapIfGreater(vec[start], vec[start + 2]); SwapIfGreater(vec[start + 1], vec[start + 2]); break;
		default: std::sort(vec.begin() + start, vec.begin() + (start + length));
		}
	}
}

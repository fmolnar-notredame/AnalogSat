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


#include "Hash.h"
#include <cstdint>

namespace analogsat
{
	// FNV-1a hash function, based on https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
	size_t fnv_hash(const unsigned char *addr, size_t count)
	{
		if (sizeof(std::size_t) == 8)
		{
			const size_t FNV_offset_basis = 14695981039346656037ULL;
			const size_t FNV_prime = 1099511628211ULL;

			size_t hash = FNV_offset_basis;
			for (size_t i = 0; i < count; i++)
			{
				hash ^= (size_t)addr[i];
				hash *= FNV_prime;
			}

			hash ^= hash >> 32;
			return hash;

		}
		else if (sizeof(std::size_t) == 4)
		{
			const size_t FNV_offset_basis = 2166136261U;
			const size_t FNV_prime = 16777619U;

			size_t hash = FNV_offset_basis;
			for (size_t i = 0; i < count; i++)
			{
				hash ^= (size_t)addr[i];
				hash *= FNV_prime;
			}
			return hash;
		}

	}
}
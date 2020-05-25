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


#ifndef ANALOGSAT_VECTORHASH_H
#define ANALOGSAT_VECTORHASH_H

#include "Hash.h"

namespace analogsat
{
	//tools to enable hashing and comparison of a vector of things, allocated sequentially in memory (implemented via sequence-equal)
	//type T must support equality comparison

	template<class T>
	class VectorHash;

	template<class T>
	class VectorCompare;

	template <class T>
	class VectorWrap
	{
		friend class VectorHash<T>;
		friend class VectorCompare<T>;

	private:
		const T* begin;
		const T* end;
		size_t hash;

	public:
		VectorWrap(const T* _begin, const T* _end)
		{
			begin = _begin;
			end = _end;
			hash = fnv_hash((const unsigned char*)begin, end - begin);
		}

		VectorWrap(const VectorWrap<T>& other)
		{
			begin = other.begin;
			end = other.end;
		}

		VectorWrap<T>& operator=(const VectorWrap<T>& other)
		{
			begin = other.begin;
			end = other.end;
			return *this;
		}
	};

	template<class T>
	class VectorHash
	{
	public:
		size_t operator()(const VectorWrap<T>& item) const
		{
			return item.hash;
		}	
	};

	template<class T>
	class VectorCompare
	{
	public:
		bool operator() (const VectorWrap<T>& obj1, const VectorWrap<T>& obj2) const
		{
			const T* b1 = obj1.begin;
			const T* b2 = obj2.begin;
			const T* e1 = obj1.end;
			const T* e2 = obj2.end;

			if (e1 - b1 != e2 - b2) return false; //different lengths

			while (b1 != e1)
			{
				if (*b1 != *b2) return false;
				++b1;
				++b2;
			}
			return true;
		}
	};

}

#endif

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


#ifndef ANALOGSAT_CUDA_STRUCTS_H
#define ANALOGSAT_CUDA_STRUCTS_H

namespace analogsat
{
	//common kernel arguments for CudaSat implementations
	template <typename TFloat>
	struct CudaSatArgs
	{
		int N, M, STRIDE;	//problem dimensions
		TFloat KNORM;		//clause normalization factor
		TFloat ALPHA;		//N/M
		TFloat B;			//bias parameter B (to be set every time before the kernel is called)

		int* GC;			//C matrix gpu ptr
		TFloat* MEAN_AM;	//mean am gpu ptr
	};
}

#endif

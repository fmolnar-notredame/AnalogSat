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


#ifndef ANALOGSAT_REDUCTION_H
#define ANALOGSAT_REDUCTION_H

#include "CudaUtils.h"

namespace analogsat
{
	//max reduction - d_odata and d_temp must be at least MAX_BLOCK_DIM_SIZE
	template<typename TFloat>
	void CudaReduceMax(int size, TFloat *d_idata, TFloat *d_odata, TFloat* d_temp);

	//sum reduction - d_odata and d_temp must be at least MAX_BLOCK_DIM_SIZE
	template<typename TFloat>
	void CudaReduceSum(int size, TFloat *d_idata, TFloat *d_odata, TFloat* d_temp);

	//boolean AND reduction - d_odata and d_temp must be at least MAX_BLOCK_DIM_SIZE
	void CudaReduceAnd(int size, int *d_idata, int *d_odata, int* d_temp);
}

#endif




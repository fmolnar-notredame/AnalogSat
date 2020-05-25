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
//along with this program. If not, see <http://www.gnu.org/licenses/>.// Method to get the GPU name

#include "gpuidtool.h"
#include <cstring>

// not needed, nvcc adds this automatically, but VS intellisense works better this way
#include <cuda.h> 
#include <cuda_runtime.h>

using namespace std;

namespace analogsat
{
	string GetGpuName()
	{
		//get current device number
		int i;
		cudaGetDevice(&i);

		//get its properties
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		return string(prop.name);
	}
}
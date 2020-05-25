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


#include "CudaUtils.h"
#include <string>
#include <cstdio>
#include <stdexcept>

using namespace std;

namespace analogsat
{
	void CudaSafe(cudaError_t result, const char* message)
	{
		if (result != cudaSuccess)
		{
			string errorMsg = "CUDA Error: ";
			errorMsg += cudaGetErrorString(result);

			if (message != NULL)
			{
				errorMsg += "\nmessage: ";
				errorMsg += message;
			}
			errorMsg += "\n";

			fprintf(stderr, errorMsg.c_str());
			throw runtime_error(errorMsg);
		}
	}

	void CurandSafe(curandStatus_t result, const char* message)
	{
		if (result != CURAND_STATUS_SUCCESS)
		{
			string errorMsg = "cuRand Error: ";
			switch (result)
			{
			case CURAND_STATUS_VERSION_MISMATCH: errorMsg += "Header file and linked library version do not match"; break;
			case CURAND_STATUS_NOT_INITIALIZED: errorMsg += "Generator not initialized"; break;
			case CURAND_STATUS_ALLOCATION_FAILED: errorMsg += "Memory allocation failed"; break;
			case CURAND_STATUS_TYPE_ERROR: errorMsg += "Generator is wrong type"; break;
			case CURAND_STATUS_OUT_OF_RANGE: errorMsg += "Argument out of range"; break;
			case CURAND_STATUS_LENGTH_NOT_MULTIPLE: errorMsg += "Length requested is not a multple of dimension"; break;
			case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: errorMsg += "GPU does not have double precision required by MRG32k3a"; break;
			case CURAND_STATUS_LAUNCH_FAILURE: errorMsg += "Kernel launch failure"; break;
			case CURAND_STATUS_PREEXISTING_FAILURE: errorMsg += "Preexisting failure on library entry"; break;
			case CURAND_STATUS_INITIALIZATION_FAILED: errorMsg += "Initialization of CUDA failed"; break;
			case CURAND_STATUS_ARCH_MISMATCH: errorMsg += "Architecture mismatch, GPU does not support requested feature"; break;
			case CURAND_STATUS_INTERNAL_ERROR: errorMsg += "Internal library error"; break;
			default: errorMsg += "Unknown cuRand error";
			}

			if (message != NULL)
			{
				errorMsg += "\nmessage: ";
				errorMsg += message;
			}
			errorMsg += "\n";

			fprintf(stderr, errorMsg.c_str());
			throw runtime_error(errorMsg);
		}
	}

}

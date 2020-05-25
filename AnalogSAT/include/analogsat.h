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


#ifndef ANALOGSAT_TOP_H
#define ANALOGSAT_TOP_H

//includes all the necessary headers of AnalogSAT

//SAT state vector
#include "../solver/SatState.h"

//SAT solving ODE systems
#include "../cpu_sat/CpuSat.h"
#include "../cpu_sat/CpuSatTanh.h"

#include "../cuda_sat/CudaSat1.h"
#include "../cuda_sat/CudaSat2.h"
#include "../cuda_sat/CudaSat3.h"

#include "../cuda_sat/CudaSatTanh1.h"
#include "../cuda_sat/CudaSatTanh2.h"
#include "../cuda_sat/CudaSatTanh3.h"

//SAT solver backend
#include "../solver/SatSolver.h"

//IO
#include "../io/CnfReader.h"
#include "../io/CnfWriter.h"

//Random interface
#include "../cpu_sat/CpuRandom.h"
#include "../cuda_base/CudaRandom.h"

//ODE integrator
#include "../cpu_sat/CpuRungeKutta.h"
#include "../cuda_sat/CudaRungeKutta.h"

//Problems - storing and making
#include "../problem/SatProblem.h"
#include "../problem/RandomSatMaker.h"
#include "../problem/PlantedSatMaker.h"
#include "../problem/FastSatMaker.h"

//use this define before including analogsat.h to enable inclusion of AnalogSAT's low level utilities
#ifdef ANALOGSAT_INCLUDE_UTILS
#include "../util/utils.h"
#include "../util/path.h"
#include "../util/cpuidtool.h"
#include "../util/gpuidtool.h"
#include "../util/Hash.h"
#include "../util/VectorHash.h"
#include "../util/Wallclock.h"
#endif

#endif

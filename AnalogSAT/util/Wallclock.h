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


#ifndef ANALOGSAT_WALLCLOCK_H
#define ANALOGSAT_WALLCLOCK_H

#include <chrono>

namespace analogsat
{
	//a simple stopwatch based on std::chrono::high_resolution_clock
	class WallClock
	{
	private:

		typedef decltype(std::chrono::high_resolution_clock::now()) clock_type;
		clock_type wallstart;
		std::chrono::duration<float> duration;
		
		bool running;
		double total_seconds;
		double TakeReading();


	public:

		WallClock();

		//start the clock
		void Start();
		
		//stop the clock (add time since last start to the total elapsed time)
		void Stop();

		//stop the clock if running, and set the total elapsed time to zero
		void Reset();

		//check if the clock is running now
		bool IsRunning() const;

		//get the total elapsed time
		double GetTotalElapsedTime();
	};
}


#endif
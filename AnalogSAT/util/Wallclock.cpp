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


#include "Wallclock.h"

namespace analogsat
{
	double WallClock::TakeReading()
	{
		clock_type wallnow = std::chrono::high_resolution_clock::now();
		duration = wallnow - wallstart;
		double elapsed_since_start = duration.count();
		return elapsed_since_start;
	}

	double WallClock::GetTotalElapsedTime()
	{
		if (running) return total_seconds + TakeReading();
		else return total_seconds;
	}

	WallClock::WallClock()
	{
		Reset();
	}

	void WallClock::Reset()
	{
		total_seconds = 0.0;
		running = false;
	}

	void WallClock::Start()
	{
		wallstart = std::chrono::high_resolution_clock::now();
		running = true;
	}

	void WallClock::Stop()
	{
		double elapsed_since_start = TakeReading();
		total_seconds += elapsed_since_start;
		running = false;
	}

	bool WallClock::IsRunning() const
	{
		return running;
	}
}
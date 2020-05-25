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


#ifndef UTILS_PATH_H
#define UTILS_PATH_H

#include <string>
#include <vector>

namespace analogsat
{
	extern const std::string pathSep;

	bool DirectoryExists(const char* path);
	bool DirectoryExists(const std::string& path);

	void CreateDir(const char* path);
	void CreateDir(const std::string& path);

	bool FileExists(const char* filename);
	bool FileExists(const std::string& filename);

	std::string GetFileNameWithoutExtension(const char* filename);
	std::string GetFileNameWithoutExtension(const std::string& filename);

	std::string GetFileNameWithoutPath(const std::string& filename);
	std::string GetFileNameWithoutPath(const char* filename);

	// break down a folder path to successive elements
	std::vector<std::string> GetPathSuccessive(const char* folder);
	std::vector<std::string> GetPathSuccessive(const std::string& folder);

}
#endif


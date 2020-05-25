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


#define _CRT_SECURE_NO_WARNINGS 1  //pacify Visual Studio

#include "path.h"
#include <string>
#include <cstdio>
#include <vector>

using namespace std;

#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#endif

namespace analogsat
{

#ifdef _WIN32
	const string pathSep("\\");
#include <Windows.h>
#else
	const string pathSep("/");
#endif

	bool DirectoryExists(const string& path) { return DirectoryExists(path.c_str()); }
	bool DirectoryExists(const char* path)
	{
#if defined _WIN32
		DWORD ftyp = GetFileAttributesA(path);
		if (ftyp == INVALID_FILE_ATTRIBUTES)
			return false;  //something is wrong with your path!

		if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
			return true;   // this is a directory!

		return false;    // this is not a directory!
#else

		struct stat info;

		if( stat( path, &info ) != 0 )
			//printf( "cannot access %s\n", pathname );
			return false;
		else if (info.st_mode & S_IFDIR) 
			//printf( "%s is a directory\n", pathname );
			return true;
		else
			//printf( "%s is no directory\n", pathname );	
			return false;
#endif
	}

	void CreateDir(const string& path) { CreateDir(path.c_str()); }
	void CreateDir(const char* path)
	{		
#if defined _MSC_VER
		_mkdir(path);
#elif defined __GNUC__
		mkdir(path, 0777);
#endif
	}


	bool FileExists(const string& filename) { return FileExists(filename.c_str()); }
	bool FileExists(const char* filename)
	{
		FILE *f = fopen(filename, "r");
		if (f == NULL) return false;
		else
		{
			fclose(f);
			return true;
		}
	}

	string GetFileNameWithoutExtension(const string& filename) { return GetFileNameWithoutExtension(filename.c_str()); }
	string GetFileNameWithoutExtension(const char* filename)
	{
		string path(filename);
		auto found = path.find_last_of(".");
		if (found == string::npos) return path;
		else return path.substr(0, found);
	}

	string GetFileNameWithoutPath(const string& filename){ return GetFileNameWithoutPath(filename.c_str()); }
	string GetFileNameWithoutPath(const char* filename)
	{
		string path(filename);
		auto found = path.find_last_of("\\/");
		return path.substr(found + 1);
	}

	// break down a folder path to successive elements that are not relative (so they can be mkdir'ed)
	vector<string> GetPathSuccessive(const string& folder) { return GetPathSuccessive(folder.c_str()); }
	vector<string> GetPathSuccessive(const char* folder)
	{
		//split the path
		string str(folder);
		vector<string> parts;
		std::size_t current, previous = 0;
		current = str.find_first_of("\\/");
		while (current != std::string::npos) {
			parts.push_back(str.substr(previous, current - previous));
			previous = current + 1;
			current = str.find_first_of("\\/", previous);
		}
		parts.push_back(str.substr(previous, current - previous));

		//find the first item that is not followed by a "." or ".." or empty:
		//these are part of a relative path name, the path decomposition should not split on these
		size_t start = parts.size();
		for (int i = (int)parts.size() - 1; i >= 0; i--)
		{
			if (parts[i].size() == 0 || parts[i].find_first_of(".") != string::npos)
				break;
			else start--;
		}

		//compose successively
		vector<string> results;
		if (parts.size() == 0) return results;

		//build up the relative path parts
		string full = "";
		for (std::size_t i = 0; i < start; i++)
		{
			if (i > 0) full += pathSep;
			full += parts[i];
		}

		//add the absolute name parts one by one
		for (std::size_t i = start; i < parts.size(); i++)
		{
			if (i > 0) full += pathSep;
			full += parts[i];
			results.push_back(full);
		}

		return results;
	}

}

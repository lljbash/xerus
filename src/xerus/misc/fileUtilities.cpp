// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2019 Benjamin Huber and Sebastian Wolf. 
// 
// Xerus is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
// 
// Xerus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with Xerus. If not, see <http://www.gnu.org/licenses/>.
//
// For further information on Xerus visit https://libXerus.org 
// or contact us at contact@libXerus.org.

/**
* @file
* @brief Implementation of the file utilities.
*/

#include <xerus/misc/fileUtilities.h>
#include <xerus/misc/internal.h>
#include <xerus/misc/stringUtilities.h>

#include <fstream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <boost/filesystem.hpp>
#pragma GCC diagnostic pop

namespace xerus { namespace misc {
		
	void create_directories_for_file(const std::string& _path) {
		if(_path.find('/') != std::string::npos) {
			const std::string folder = _path.substr(0, _path.find_last_of('/'));
			boost::filesystem::create_directories(folder);
		}
	}
	
	
	std::set<std::string> get_files(const std::string& _path) {
		REQUIRE(boost::filesystem::exists(_path) && boost::filesystem::is_directory(_path), "Invalid path " << _path << " (not an existing directory).");
		std::set<std::string> files;
		
		for (boost::filesystem::directory_iterator itr(_path), end; itr != end; ++itr) {
			if(!boost::filesystem::is_regular_file(itr->path())) { continue;}
			const auto file = itr->path().filename().string();
			files.emplace(file);
		}
		
		return files;
	}
	
	
	std::set<std::string> get_directories(const std::string& _path) {
		REQUIRE(boost::filesystem::exists(_path) && boost::filesystem::is_directory(_path), "Invalid path " << _path << " (not an existing directory).");
		std::set<std::string> files;
		
		for (boost::filesystem::directory_iterator itr(_path), end; itr != end; ++itr) {
			if(!boost::filesystem::is_directory(itr->path())) { continue;}
			const auto file = itr->path().filename().string();
			files.emplace(file);
		}
		
		return files;
	}
	
	
	bool file_exists(const std::string& _path) {
		return boost::filesystem::exists(_path);
	}
	
	bool file_is_empty(const std::string& _path) {
		std::ifstream pFile(_path);
		REQUIRE(pFile.is_open(), "IE: Failed to open file");
		return pFile.peek() == std::ifstream::traits_type::eof();
	}
	

	std::string read_file(const std::string& _path) {
		REQUIRE(boost::filesystem::exists(_path), "File " << _path << " does not exist.");
		std::ifstream fileStream(_path, std::ifstream::in);
		CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (read) open the file " << _path);
		
		std::string contents;
		fileStream.seekg(0, std::ios::end);
		contents.resize(size_t(fileStream.tellg()));
		fileStream.seekg(0, std::ios::beg);
		fileStream.read(&contents[0], std::streamsize(contents.size()));
		return contents;
	}
	
	
	std::string normalize_pathname(const std::string &_name) {
		std::vector<std::string> oldpath = explode(_name,'/');
		std::vector<std::string *> newpath;
		for (std::string &f : oldpath) {
			if (f.empty()) { continue; }
			if (f==".." && !newpath.empty() && *newpath.back() != "..") {
				newpath.pop_back();
			} else {
				newpath.push_back(&f);
			}
		}
		std::string ret;
		for (std::string *f : newpath) {
			ret += *f;
			ret += '/';
		}
		if (!ret.empty()) { ret.pop_back(); }
		return ret;
	}
	
	
	
	#if __cplusplus >= 201402L
	
		std::ifstream open_file_read(const std::string& _path) {
			REQUIRE(boost::filesystem::exists(_path), "File " << _path << " does not exist.");
			std::ifstream fileStream(_path, std::ifstream::in);
			CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (read) open the file " << _path);
			return fileStream;
		}
		
		
		std::ofstream open_file_truncate(const std::string& _path) {
			create_directories_for_file(_path);
			std::ofstream fileStream(_path, std::ofstream::out | std::ofstream::trunc);
			CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (write) open the file " << _path);
			return fileStream;
		}
		
		
		std::ofstream open_file_append(const std::string& _path) {
			create_directories_for_file(_path);
			std::ofstream fileStream(_path, std::ofstream::out | std::ofstream::app);
			CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (write) open the file " << _path);
			return fileStream;
		}
	
	#endif
} }



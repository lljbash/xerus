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
* @brief Header file for some minimalistic file utilities.
*/

#pragma once

#include <set>
#include <string>
#include <fstream>

namespace xerus { namespace misc {
	
	///@brief Creates all directories in the @a _path of the file, if needed.
	void create_directories_for_file(const std::string& _path);
	
	///@brief Returns a set containing all regular files at the given @a _path.
	std::set<std::string> get_files(const std::string& _path);

	///@brief Checks whether a file at the given @a _path exists.
	bool file_exists(const std::string& _path);
	
	///@brief Checks whether a file at the given @a _path is empty.
	bool file_is_empty(const std::string& _path);
	
	///@brief Reads the complete content of the file at the given @a _path into a string.
	std::string read_file(const std::string& _path);
	
	
	#if __cplusplus >= 201402L

	///@brief Opens a reading filestream for the given @a _path.
	std::ifstream open_file_read(const std::string& _path);

	///@brief Opens a writing filestream for the given @a _path, deleting any preexisting content.
	std::ofstream open_file_truncate(const std::string& _path);

	///@brief Opens a writing filestream for the given @a _path, appending to any preexisting content.
	std::ofstream open_file_append(const std::string& _path);
	
	#endif
} }

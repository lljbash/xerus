// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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
    
    void create_folder_for_file(const std::string& _path);
    
    std::set<std::string> get_files(const std::string& _path);

    bool file_is_empty(const std::string& _filename);

    std::string read_file(const std::string& _path);
    
    #if __cplusplus <= 201103L
        #warning Some xerus file utilities require a C++14 compliant compiler
    #else
   
    void open_file_read(std::ifstream &_fileStream, const std::string &_path);

    void open_file_truncate(std::ofstream &_fileStream, const std::string &_path);

    void open_file_append(std::ofstream &_fileStream, const std::string &_path);
    
    #endif
} }




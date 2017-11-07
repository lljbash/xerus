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
* @brief Implementation of the file utilities.
*/

#include <xerus/misc/fileUtilities.h>
#include <xerus/misc/internal.h>

#include <fstream>
#include <boost/filesystem.hpp>

namespace xerus { namespace misc {
        
    void create_folder_for_file(const std::string& _path) {
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
    
    bool file_is_empty(const std::string& _filename) {
        std::ifstream pFile(_filename);
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
    
    
    
    #if __cplusplus >= 201402L
    
        std::ifstream open_file_read(const std::string& _path) {
            REQUIRE(boost::filesystem::exists(_path), "File " << _path << " does not exist.");
            std::ifstream fileStream(_path, std::ifstream::in);
            CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (read) open the file " << _path);
            return fileStream;
        }
        
        
        std::ofstream open_file_truncate(const std::string& _path) {
            create_folder_for_file(_path);
            std::ofstream fileStream(_path, std::ofstream::out | std::ofstream::trunc);
            CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (write) open the file " << _path);
            return fileStream;
        }
        
        
        std::ofstream open_file_append(const std::string& _path) {
            create_folder_for_file(_path);
            std::ofstream fileStream(_path, std::ofstream::out | std::ofstream::app);
            CHECK(fileStream.is_open() && !fileStream.fail(), error, "Could not properly (write) open the file " << _path);
            return fileStream;
        }
    
    #endif
} }



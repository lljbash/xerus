// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf. 
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
* @brief Header file for some helper functions.
*/

#pragma once

#include <string>

namespace xerus {
	namespace misc {
		/**
		* @brief Execute a given command.
		* @param _cmd the command to execute
		* @return the cout return of the command.
		*/
		std::string exec(const std::string &_cmd);
		
		
		/**
		* @brief Execute a given command and pipe _stdin to its std input,
		* @param _cmd the command to execute
		* @param _stdin the input for the program
		*/
		void exec(const std::string & _cmd, const std::string &_stdin);
	}
}

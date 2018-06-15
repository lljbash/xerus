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
* @brief Header file for the call-stack functionality.
*/

#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace xerus {
	namespace misc {
		/**
		* @brief Returns a string representation of the current call-stack (excluding the function itself).
		* @details Per default this uses the binutils library to get the following information:
		* [address .section] filename:line (function)
		* if all of these are available.
		*/
		std::string get_call_stack();

		/**
		* @brief Returns the address range of the elf-section names @a _name as part of the executable / so file that contained @a _addr.
		*/
		std::pair<uintptr_t, uintptr_t> get_range_of_section(void* _addr, std::string _name);
	}
}

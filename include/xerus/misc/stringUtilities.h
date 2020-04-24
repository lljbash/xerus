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
 * @brief Header file for some elementary string manipulation routines.
 */

#pragma once

#include <string>
#include <vector>

#include "standard.h"

namespace xerus {
	namespace misc {
		/// @brief Demangles the function and class names created by gcc into a more readable format.
		std::string XERUS_warn_unused demangle_cxa(const std::string& _cxa);

		///@brief: Explodes a string at positions indicated by _delim.
		std::vector<std::string> explode(const std::string& _string, const char _delim);
		
		///@brief: Returns true if _string contains _search as a substring.
		bool contains(const std::string& _string, const std::string& _search);

		///@brief: Replaces all occurences of _search in _string by _replace.
		void replace(std::string& _string, const std::string& _search, const std::string& _replace);
		
		///@brief: Removes all leading and trailing whitespaces from _string.
		std::string XERUS_warn_unused trim(const std::string& _string, const std::string& whitespace = " \t\n\r\v");

		///@brief: Removes all leading and trailing whitespaces from _string, and reduces all double whitespaces to one.
		std::string XERUS_warn_unused reduce(const std::string& _string, const std::string& whitespace = " \t\n\r\v", const std::string& fill = " ");
	}
}

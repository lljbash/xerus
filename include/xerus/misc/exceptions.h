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
 * @brief Header file for xerus::misc::generic_error exception class.
 */

#pragma once

#include <exception>
#include <sstream>
#include "stringFromTo.h"

namespace xerus {
    namespace misc {
		/**
		 * @brief The xerus exception class.
		 * @details All exceptions thrown by xerus are of this type (or subclasses thereof).
		 */
        class generic_error : public std::exception {
        public:
			///@brief String containing all relevant information concerning this error.
			std::string errorInfo;
            
            const char* what() const noexcept override { return errorInfo.c_str(); }
        };
		
		/// @brief The pipe operator allows to add everything that can be converted to string to the error_info and derived exceptions. 
		template<typename error_t, class T>
		typename std::enable_if<std::is_base_of<generic_error, error_t>::value, error_t&>::type
		operator<<(error_t& o, const T& _info) noexcept {
			std::ostringstream errorStream(o.errorInfo, std::ios_base::ate | std::ios_base::out);
			errorStream << _info;
			o.errorInfo = errorStream.str();
			return o;
		}
		
		/// @brief The pipe operator allows to add everything that can be converted to string to the error_info and derived exceptions. 
		template<typename error_t, class T>
		typename std::enable_if<std::is_base_of<generic_error, error_t>::value, error_t&>::type
		operator<<(error_t&& o, const T& _info) noexcept {
			std::ostringstream errorStream(o.errorInfo, std::ios_base::ate | std::ios_base::out);
			errorStream << _info;
			o.errorInfo = errorStream.str();
			return o;
		}
		
// 		Once we have C++20... :-/
// 		/// @brief The pipe operator allows to add everything that can be converted to string to the error_info and derived exceptions. 
// 		template<typename error_t, class T, typename std::enable_if<std::is_base_of<generic_error, std::remove_cvref_t<error_t>>::value, int>::type = 0>
// 		std::remove_cvref_t<error_t>& operator<<(error_t&& o, const T& _info) noexcept {
// 			std::ostringstream errorStream(o.errorInfo, std::ios_base::ate | std::ios_base::out);
// 			errorStream << _info;
// 			o.errorInfo = errorStream.str();
// 			return o;
// 		}
    }
}

/**
 * @def XERUS_THROW(...)
 * @brief Helper macro to throw a generic_error (or derived exception) with some additional information included (function name, file name and line number).
 */
#define XERUS_THROW(...) throw (__VA_ARGS__ << "\nexception thrown in function: " << (__func__) << " (" << (__FILE__) <<" : " << (__LINE__) << ")\n")

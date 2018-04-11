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
            std::string error_info;
            
			/// @brief: Normal constructor without preset error_info.
            generic_error();
            
			/// @brief Standard copy constructor.
            generic_error(const generic_error &_other) noexcept;
            
            const char* what() const noexcept override;
        };
		
		/// @brief The pipe operator allows to add everything that can be converted to string to the error_info and derived exceptions. 
		template<typename error_t, class T>
		typename std::enable_if<std::is_base_of<generic_error, error_t>::value, error_t&>::type
		operator<< (error_t &o, const T &_info) noexcept {
			o.error_info += to_string(_info);
			return o;
		}
		
		/// @brief The pipe operator allows to add everything that can be converted to string to the error_info and derived exceptions. 
		template<typename error_t, class T>
		typename std::enable_if<std::is_base_of<generic_error, error_t>::value, error_t&>::type
		operator<< (error_t &&o, const T &_info) noexcept {
			o.error_info += to_string(_info);
			return o;
		}
    }
}

/**
 * @def XERUS_THROW(...)
 * @brief Helper macro to throw a generic_error (or derived exception) with some additional information included (function name, file name and line number).
 */
#define XERUS_THROW(...) throw (__VA_ARGS__ << "\nexception thrown in function: " << (__func__) << " (" << (__FILE__) <<" : " << (__LINE__) << ")\n")

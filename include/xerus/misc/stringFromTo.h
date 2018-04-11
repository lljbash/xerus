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
 * @brief Header file for some elementary string manipulation routines.
 */

#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <utility>

#include "standard.h"
#include "sfinae.h"
#include "containerOutput.h"

namespace xerus {
	namespace misc {
		// Enable all standard to string functions also in xerus misc namespace
		using std::to_string;
		
		static XERUS_force_inline std::string to_string(const bool obj) {
			return obj ? std::string("TRUE") : std::string("FALSE");
		}
		
		static XERUS_force_inline std::string to_string(const std::string& obj) {
			return obj;
		}

		static XERUS_force_inline std::string to_string(const char * const obj) {
			return std::string(obj);
		}
		
		#if __GNUC__ > 4 || defined(__clang__)
		
			XERUS_GENERATE_EXISTS_FUNCTION(to_string)
			
			///@brief: Converts any object (which has no specialized function for it) to string.
			template<typename T, typename std::enable_if<!sfinae::exists_to_string<T>::value, int>::type = 0>
			std::string to_string(const T& obj) {
				std::ostringstream stream;
				stream << obj;
				return stream.str();
			}
			
		#else
			
			///@brief: Converts any object (which has no specialized function for it) to string.
			template<typename T>
			std::string to_string(const T& obj) {
				std::ostringstream stream;
				stream << obj;
				return stream.str();
			}
			
		#endif
		
		
		///@brief: Converts an arbitary Object to string with fixed precision
		template<typename T>
		std::string to_string(const T& obj, const long _precision) {
			std::ostringstream stream;
			stream.precision(_precision);
			stream << std::fixed << obj;
			return stream.str();
		}
		
		
		
		
		
		///@brief: Creates an arbitary Object from string
		template<typename T>
		T from_string(const std::string& _s){
			std::stringstream stream(_s);
			T tmp;
			stream >> tmp;
			return tmp;
		}
		
		template<>
		XERUS_force_inline int from_string<int>(const std::string& _str) {
			return stoi(_str);
		}
		
		template<>
		XERUS_force_inline long from_string<long>(const std::string& _str) {
			return stol(_str);
		}
		
		template<>
		XERUS_force_inline long long from_string<long long>(const std::string& _str) {
			return stoll(_str);
		}
		
		template<>
		XERUS_force_inline unsigned long from_string<unsigned long>(const std::string& _str) {
			return stoul(_str);
		}
		
		template<>
		XERUS_force_inline unsigned long long from_string<unsigned long long>(const std::string& _str) {
			return stoull(_str);
		}
		
		template<>
		XERUS_force_inline float from_string<float>(const std::string& _str) {
			return stof(_str);
		}
		
		template<>
		XERUS_force_inline double from_string<double>(const std::string& _str) {
			return stod(_str);
		}
		
		template<>
		XERUS_force_inline long double from_string<long double>(const std::string& _str) {
			return stold(_str);
		}
	}
}

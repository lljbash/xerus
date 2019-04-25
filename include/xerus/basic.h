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
 * @brief Header file for shorthand notations that are xerus specific but used throughout the library.
 */

#pragma once

#include <stddef.h>
#include "misc/standard.h"
#include "misc/namedLogger.h"
#include "tensorLogger.h"

/**
* @brief The main namespace of xerus.
* @details With very few exceptions all classes, functions and variables declared by the xerus are placed in this namespace.
*/
namespace xerus {
    /**
	 * @brief The type of values to be used by xerus.
	 * @details In future versions this should be allowed to be float, double, or complex.
	 * In the current version however this is fixed to double.
	 */
	using value_t = double;
	
	/**
	 * @brief The default epsilon value in xerus.
	 * @details This value is used as a default value by all xerus routines that work up to a certain accuracy,
	 * e.g. the singluar value decompostion or the ALS algorithm.
	 */
	constexpr const value_t EPSILON = 8*std::numeric_limits<value_t>::epsilon();
    
	/**
	* @brief Namespace for function and classes designated only for internal use.
	* @details End users should never be required to know or explicitly use classes or functions from this
	* namespace. If you want to do so nevertheless be aware of the minimalistic documentation and in some cases 
	* quite unexpected behavior.
	*/
	namespace internal {
		/// @brief Internal deleter function, needed because std::shared_ptr misses an array overload.
		void array_deleter_vt(value_t* const _toDelete);
		
		/// @brief Internal deleter function, needed because std::shared_ptr misses an array overload.
		void array_deleter_st(size_t* const _toDelete);
	}
}

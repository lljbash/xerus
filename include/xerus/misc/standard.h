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
 * @brief Header file for global shorthand notations of elementary integer types and attribute lists.
 */

#pragma once

#include <stddef.h>
#include <cstdint>
#include <cstddef>

namespace xerus {
	/**
	 * @brief Collection of classes and functions that provide elementary functionality that is not special to xerus as a tensor library.
	 */
	namespace misc {
		/// @brief Namespace for function and classes designated only for internal use.
		namespace internal {}
	}
	
	/**
	 * The version of the compiled xerus library
	 */
	extern const int VERSION_MAJOR;
	extern const int VERSION_MINOR;
	extern const int VERSION_REVISION;
	extern const int VERSION_COMMIT;

	// Shorter names for unsigned types
	using byte = uint8_t;
	using ushort = unsigned short;
	using uint = unsigned int;
	using ulong = unsigned long;

	// Shorter names for fixed width types
	using int8 = int8_t;
	using int16 = int16_t;
	using int32 = int32_t;
	using int64 = int64_t;

	using uint8 = uint8_t;
	using uint16 = uint16_t;
	using uint32 = uint32_t;
	using uint64 = uint64_t;
}

/** 
 * @def XERUS_force_inline 
 * @brief Collection of attributes to force gcc to inline a specific function.
 */
#if defined(__clang__)
	#define XERUS_force_inline  inline
#else
	#define XERUS_force_inline  inline __attribute__((always_inline, gnu_inline))
#endif

/** 
 * @def XERUS_deprecated
 * @brief Attribute to mark deprecated functions if supported by currently used c++ version.
 */
#if __cplusplus > 201400L
	#define XERUS_deprecated(msg) [[deprecated(msg)]]
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ * 100 + __GNUC_MINOR__) >= 409
	#define XERUS_deprecated(msg) [[gnu::deprecated(msg)]]
#else
	#define XERUS_deprecated(msg)
#endif

#define XERUS_warn_unused	__attribute__((warn_unused_result))

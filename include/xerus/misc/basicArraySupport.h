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
* @brief Header file for the low level array support function.
*/

#pragma once

#include <type_traits>
#include <memory>
#include <cstring>

#include "standard.h"

namespace xerus {
	namespace misc {
		template<class T>
		std::unique_ptr<T[]> make_unique_array(T* _ptr) {
			return std::unique_ptr<T[]>(_ptr);
		}
		
		/** 
		* @brief Sets all entries equal to zero.
		* @details This is done directly by memset and therefore requires the type to be trivial.
		*/
		template <typename T, typename std::enable_if<std::is_trivial<T>::value, int>::type = 0>
		inline void set_zero(T* const __restrict _x, const size_t _n) noexcept {
			memset(_x, 0, _n*sizeof(T));
		}
		
		
		/** 
		* @brief Copys @a _n entries from @a _y to @a _x, where @a _y and @a _x must be disjunkt memory regions.
		* @details For trivial copyable types this is only a slim wrapper for memcpy.
		*/
		template <typename T, typename std::enable_if<std::is_trivial<T>::value, int>::type = 0>
		inline void copy(T* const __restrict _dest, const T* const __restrict _src, const size_t _n) noexcept {
			memcpy(_dest, _src, _n*sizeof(T));
		}
		
		
		/** 
		* @brief Copys @a _n entries from @a _y to @a _x, allowing the accessed memry regions of @a _y and @a _x to overlap.
		* @details For trivial copyable types this is only a slim wrapper for memmove.
		*/
		template <typename T, typename std::enable_if<std::is_trivial<T>::value, int>::type = 0>
		inline void copy_inplace(T* const _x, const T* const _y, const size_t _n) noexcept {
			memmove(_x, _y, _n*sizeof(T));
		}
		
		
		/** 
		* @brief Copys @a _n entries from @a _y to @a _x, simulationously scaling each entry by the factor @a _alpha. I.e x = alpha*y.
		*/
		template <typename T>
		inline void copy_scaled(T* const __restrict _x, const T _alpha, const T* const _y, const size_t _n) noexcept {
			for(size_t i = 0; i < _n; ++i) {
				_x[i] = _alpha*_y[i];
			}
		}
		
		
		/** 
		* @brief Scales @a _n entries of @a _x by the factor @a _alpha. I.e. x = alpha*x.
		*/
		template <typename T>
		inline void scale(T* const __restrict _x, const T _alpha, const size_t _n) noexcept {
			for(size_t i = 0; i < _n; i++) {
				_x[i] *= _alpha;
			}
		}
		
		
		/** 
		* @brief Adds @a _n entries of @a _y to the ones of @a _x. I.e. x += y.
		*/
		template <typename T>
		inline void add(T* const __restrict _x, const T* const __restrict _y, const size_t _n) noexcept {
			for(size_t i = 0; i < _n; i++) {
				_x[i] += _y[i];
			}
		}
		
		
		/** 
		* @brief Adds @a _n entries of @a _y, scaled by @a _alpha to the ones of @a _x. I.e. x += alpha*y.
		*/
		template <typename T>
		inline void add_scaled(T* const __restrict _x, const T _alpha, const T* const __restrict _y, const size_t _n) noexcept {
			for(size_t i = 0; i < _n; i++) {
				_x[i] += _alpha*_y[i];
			}
		}
	}
}

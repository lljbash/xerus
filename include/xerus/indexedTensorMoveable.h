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
 * @brief Header file for the IndexedTensorMoveable class.
 */

#pragma once

#include "indexedTensorWritable.h"
#include <type_traits>

namespace xerus {
	namespace internal {
		/**
		* @brief Internal representation of an read and write and moveable indexed Tensor or TensorNetwork.
		* @details This class appears inplicitly by indexing any Tensor or TensorNetwork. It is not recommended to use
		* it explicitly or to store variables of this type (unless you really know what you are doing).
		*/
		template<class tensor_type>
		class IndexedTensorMoveable final : public IndexedTensorWritable<tensor_type> {
		public:
			///@brief Creates an empty indexed Tensor, should only be used internally
			IndexedTensorMoveable();
			
			///@brief There is no usefull copy constructor, because the handling of the tensorObject is unclear.
			IndexedTensorMoveable(const IndexedTensorMoveable &_other ) = delete;
			
			///@brief Move constructor
			IndexedTensorMoveable(      IndexedTensorMoveable &&_other ) noexcept;
			
			///@brief Constructs an IndexedTensorMoveable with the given tensor and indices and if ordered to do so takes ownership of the tensorObject.
			IndexedTensorMoveable(tensor_type* const _tensorObject, const std::vector<Index>& _indices);
			
			///@brief Constructs an IndexedTensorMoveable with the given tensor and indices and if ordered to do so takes ownership of the tensorObject
			IndexedTensorMoveable(tensor_type* const _tensorObject,       std::vector<Index>&& _indices);
			
			///@brief Allow explicit cast from IndexedTensorReadOnly.
			explicit IndexedTensorMoveable(IndexedTensorReadOnly<tensor_type>&& _other);
			
			///@brief Allow explicit conversions from indexed TensorNetworks to indexed Tensors
			template<class X = tensor_type, typename std::enable_if<std::is_base_of<Tensor, typename std::decay<X>::type>::value, int>::type = 0>
			explicit IndexedTensorMoveable(IndexedTensorReadOnly<TensorNetwork>&& _other );
			
			///@brief Allow implicit conversions from indexed Tensors to indexed TensorNetworks
			#if __GNUC__ > 4
			template<class X = tensor_type, typename std::enable_if<std::is_base_of<TensorNetwork, typename std::decay<X>::type>::value, int>::type = 0>
			#else
			template<class X = tensor_type>
			#endif
			IndexedTensorMoveable(IndexedTensorReadOnly<Tensor>&& _other);
			
			#if __GNUC__ > 4
			template<class X = tensor_type, typename std::enable_if<std::is_base_of<TensorNetwork, typename std::decay<X>::type>::value, int>::type = 0>
			#else
			template<class X = tensor_type>
			#endif
			IndexedTensorMoveable(IndexedTensorReadOnly<Tensor>& _other) : IndexedTensorMoveable(std::move(_other)) {}
		};
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Aritmetic Operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		template<class tensor_type>
		inline IndexedTensorMoveable<tensor_type> operator*(const value_t _factor, IndexedTensorMoveable<tensor_type>&& _tensor) {
			IndexedTensorMoveable<tensor_type> result(std::move(_tensor));
			*result.tensorObject *= _factor;
			return result;
		}
		
		template<class tensor_type>
		inline IndexedTensorMoveable<tensor_type> operator*(IndexedTensorMoveable<tensor_type>&& _tensor, const value_t _factor) {
			return operator*(_factor, std::move(_tensor));
		}
		
		template<class tensor_type>
		inline IndexedTensorMoveable<tensor_type> operator/(IndexedTensorMoveable<tensor_type>&& _tensor, const value_t _divisor) {
			IndexedTensorMoveable<tensor_type> result(std::move(_tensor));
			*result.tensorObject /= _divisor;
			return result;
		}
	}
}

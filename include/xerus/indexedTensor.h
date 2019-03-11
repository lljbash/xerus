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
 * @brief Header file for the IndexedTensor class.
 */

#pragma once

#include "indexedTensorWritable.h"

namespace xerus {
	namespace internal {
		template<class tensor_type>
		/**
		* @brief Internal representation of an readable and writeable indexed Tensor or TensorNetwork.
		* @details This class appears inplicitly by indexing any Tensor or TensorNetwork. It is not recommended to use
		* it explicitly or to store variables of this type (unless you really know what you are doing).
		*/
		class IndexedTensor final : public IndexedTensorWritable<tensor_type> {
		public:     
			///@brief There is no usefull copy constructor, because the handling of the tensorObject is unclear.
			IndexedTensor(const IndexedTensor &_other ) = delete;
			
			///@brief Move constructor
			IndexedTensor(IndexedTensor &&_other );
			
			///@brief Constructs an IndexedTensor with the given tensor and indices and if ordered to do so takes owership of the tensorObject
			IndexedTensor(tensor_type* const _tensorObject, const std::vector<Index>&  _indices, const bool _takeOwnership);
			
			///@brief Constructs an IndexedTensor with the given tensor and indices and if ordered to do so takes owership of the tensorObject
			IndexedTensor(tensor_type* const _tensorObject,       std::vector<Index>&& _indices, const bool _takeOwnership);
				
			/**
			* @brief Assignment operators -- Used for tensor assignment WITH indices.
			* @details Note that this is NOT a classical assignment operator. In particular this and _rhs are NOT equivalent after
			* the assignment.
			*/
			void operator=(IndexedTensorReadOnly<Tensor>&& _rhs);
			
			/**
			* @brief Assignment operators -- Used for tensor assignment WITH indices.
			* @details Note that this is NOT a classical assignment operator. In particular this and _rhs are NOT equivalent after
			* the assignment.
			*/
			void operator=(IndexedTensorReadOnly<TensorNetwork>&& _rhs);
			
			/**
			* @brief Tensor add_assignment with indices.
			*/
			void operator+=(IndexedTensorReadOnly<tensor_type>&& _rhs);
			
			/**
			* @brief Tensor subtract_assignment with indices.
			*/
			void operator-=(IndexedTensorReadOnly<tensor_type>&& _rhs);
			
			///@brief The following would be deleted due to move constructor and is therefore implemented here, calls the IndexedTensorReadOnly version. 
			void operator=(IndexedTensor<tensor_type>&& _rhs);
		};
	}
}
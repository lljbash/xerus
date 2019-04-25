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
 * @brief Header file for the IndexedTensorReadOnly class.
 */

#pragma once

#include "basic.h"
#include <vector>

namespace xerus {
	// Necessary forward declaritons
	class Index;
	class Tensor;
	class TensorNetwork;
	
	namespace internal {
		// Necessary forward declaritons
		template<class tensor_type> class IndexedTensorWritable;
		template<class tensor_type> class IndexedTensorMoveable;
		template<class tensor_type> class IndexedTensor;

		
		/**
		* @brief Internal representation of an readable indexed Tensor or TensorNetwork.
		* @details This class appears inplicitly by indexing any Tensor or TensorNetwork. It is not recommended to use
		* it explicitly or to store variables of this type (unless you really know what you are doing).
		*/
		template<class tensor_type>
		class IndexedTensorReadOnly {
		public:
			/// @brief Pointer to the associated Tensor/TensorNetwork object.
			const tensor_type* tensorObjectReadOnly;
			
			/// @brief Vector of the associates indices.
			std::vector<Index> indices;
			
			/// @brief Flag indicating whether the indices are assinged.
			bool indicesAssigned = false;
			
			/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

			/// @brief There is no usefull default constructor
			IndexedTensorReadOnly() = delete;
			
			/// @brief There is no usefull copy constructor for IndexedTensors.
			IndexedTensorReadOnly(const IndexedTensorReadOnly& _other ) = delete;
			
			/// @brief Move-constructor
			IndexedTensorReadOnly(IndexedTensorReadOnly<tensor_type> && _other ) noexcept;
			
			/// @brief Constructs an IndexedTensorReadOnly using the given pointer and indices.
			IndexedTensorReadOnly(const tensor_type* const _tensorObjectReadOnly, std::vector<Index> _indices);
			
			/// @brief Destructor must be virtual
			virtual ~IndexedTensorReadOnly();
			
			
			/*- - - - - - - - - - - - - - - - - - - - - - - - - - Standard operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
			
			///@brief: IndexedTensorReadOnly cannot be assigned as they are read only.
			void operator=(const IndexedTensorReadOnly&  _rhs) = delete;
			
			///@brief: IndexedTensorReadOnly cannot be assigned as they are read only.
			void operator=(      IndexedTensorReadOnly&& _rhs) = delete;
			
			
			/*- - - - - - - - - - - - - - - - - - - - - - - - - - Others - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
			
			///@brief Allows cast to value_t if the order of the current object is equal to 0.
			explicit operator value_t() const;
			
			///@brief Checks whether _otherTensor is the tensorObejct of this IndexTensor.
			bool uses_tensor(const tensor_type* _otherTensor) const;
			
			///@brief Returns the order of the associated tensorObejct
			size_t order() const;
			
			///@brief Assignes the indices using the order of the tensorObejct.
			void assign_indices();
			
			///@brief Assignes the indices assuming the given order.
			void assign_indices(const size_t _order);
			
			///@brief Assignes the indices using the current tensorObejct.
			void assign_index_dimensions();
			
			///@brief Returns the dimensionTuple the evaluation of this IndexedTensor to the given indices would have.
			std::vector<size_t> get_evaluated_dimensions(const std::vector<Index>& _indexOrder);
		};
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Aritmetic Operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		template<class tensor_type>
		IndexedTensorMoveable<tensor_type> operator*(const value_t _factor, IndexedTensorReadOnly<tensor_type>&& _tensor);
		
		template<class tensor_type>
		IndexedTensorMoveable<tensor_type> operator*(IndexedTensorReadOnly<tensor_type>&&_tensor, const value_t _factor);
		
		template<class tensor_type>
		IndexedTensorMoveable<tensor_type> operator/(IndexedTensorReadOnly<tensor_type>&& _tensor, const value_t _divisor);
		
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorMoveable<Tensor>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorMoveable<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorMoveable<Tensor>&& _lhs, IndexedTensorMoveable<Tensor>&& _rhs);
		
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorMoveable<Tensor>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorMoveable<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorMoveable<Tensor>&& _lhs, IndexedTensorMoveable<Tensor>&& _rhs);
		
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs);
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs);
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs);
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs);
		
		
		IndexedTensorMoveable<TensorNetwork> operator+(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs);
		IndexedTensorMoveable<TensorNetwork> operator-(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs);
		
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs);
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorMoveable<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs);
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorMoveable<TensorNetwork>&& _rhs);
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorMoveable<TensorNetwork>&& _lhs, IndexedTensorMoveable<TensorNetwork>&& _rhs);
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Higher functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		
		
		///@brief Returns the frobenious norm of the associated tensorObejct.
		template<class tensor_type>
		value_t frob_norm(const IndexedTensorReadOnly<tensor_type>& _idxTensor);
		
		///@brief Returns the one-norm of the associated tensor Obejct.
		value_t one_norm(const IndexedTensorReadOnly<Tensor>& _idxTensor);
		
		size_t get_eval_order(const std::vector<Index>& _indices);
	}
	
	
	internal::IndexedTensorMoveable<Tensor> operator/ (internal::IndexedTensorReadOnly<Tensor>&& _b, internal::IndexedTensorReadOnly<Tensor>&& _A);
	
	
	void solve(internal::IndexedTensorWritable<Tensor>&& _x, internal::IndexedTensorReadOnly<Tensor>&& _A, internal::IndexedTensorReadOnly<Tensor>&& _b);
}

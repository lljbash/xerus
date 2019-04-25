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
 * @brief Implementation of the IndexedTensorReadOnly class.
 */

#include <xerus/indexedTensorReadOnly.h>
#include <xerus/indexedTensorMoveable.h>

#include <xerus/index.h>
 
#include <xerus/misc/containerSupport.h>
#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>
#include <xerus/tensor.h>
#include <xerus/tensorNetwork.h>

namespace xerus {
	namespace internal {
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

		template<class tensor_type>
		IndexedTensorReadOnly<tensor_type>::IndexedTensorReadOnly(IndexedTensorReadOnly<tensor_type> && _other ) noexcept :
			tensorObjectReadOnly(_other.tensorObjectReadOnly),
			indices(std::move(_other.indices))
			{ }
		
		
		template<class tensor_type>
		IndexedTensorReadOnly<tensor_type>::IndexedTensorReadOnly(const tensor_type* const _tensorObjectReadOnly, std::vector<Index> _indices)
			: tensorObjectReadOnly(_tensorObjectReadOnly), indices(std::move(_indices)) { }
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Destructor - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		template<class tensor_type>
		IndexedTensorReadOnly<tensor_type>::~IndexedTensorReadOnly() = default;
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Others - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		template<class tensor_type>
		IndexedTensorReadOnly<tensor_type>::operator value_t() const {
			REQUIRE(order() == 0, "cannot cast tensors of order > 0 to value_t. did you mean frob_norm() or similar?");
			return (*tensorObjectReadOnly)[0];
		}
		
		template<class tensor_type>
		bool IndexedTensorReadOnly<tensor_type>::uses_tensor(const tensor_type *_otherTensor) const {
			return _otherTensor == tensorObjectReadOnly;
		}
		
		template<class tensor_type>
		size_t IndexedTensorReadOnly<tensor_type>::order() const {
			return tensorObjectReadOnly->order();
		}
		
		template<class tensor_type>
		void IndexedTensorReadOnly<tensor_type>::assign_indices() {
			assign_indices(order());
		}
		
		template<class tensor_type>
		void IndexedTensorReadOnly<tensor_type>::assign_indices(const size_t _order) {
			if(!indicesAssigned) {
				size_t dimensionCount = 0;
				for(size_t i = 0; i < indices.size(); ++i) {
					Index& idx = indices[i];
					
					// Set span
					idx.set_span(_order);
					
					dimensionCount += idx.span;
					
					if(!idx.fixed()) {
						// Determine whether the index is open
						bool open = true;
						for(size_t j = 0; j < i; ++j) {
							if(indices[j] == idx) {
								REQUIRE(indices[j].open(), "An index must not appere more than twice!");
								indices[j].open(false);
								open = false;
								break;
							}
						}
						
						if(open) { idx.open(true); }
					}
					IF_CHECK(idx.flags[Index::Flag::ASSINGED] = true;)
				}
				
				REQUIRE(dimensionCount >= _order, "Order determined by Indices is to small. Order according to the indices " << dimensionCount << ", according to the tensor " << _order);
				REQUIRE(dimensionCount <= _order, "Order determined by Indices is to large. Order according to the indices " << dimensionCount << ", according to the tensor " << _order);

				misc::erase(indices, [](const Index& _idx) { return _idx.span == 0; });
				indicesAssigned = true;
			}
		}
		
		template<class tensor_type>
		void IndexedTensorReadOnly<tensor_type>::assign_index_dimensions() {
			INTERNAL_CHECK(indicesAssigned, "bla");
			
			size_t dimensionCount = 0;
			for(size_t i = 0; i < indices.size(); ++i) {
				Index& idx = indices[i];
				
				// Calculate multDimension
				REQUIRE(dimensionCount+idx.span <= tensorObjectReadOnly->dimensions.size(), "Order determined by Indices is to large: " << dimensionCount+idx.span << " > " << tensorObjectReadOnly->dimensions.size());
				idx.assingedDimension = 1;
				for(size_t j = 0; j < idx.span; ++j) {
					idx.assingedDimension *= tensorObjectReadOnly->dimensions[dimensionCount++];
				}
			}
			
			REQUIRE(dimensionCount >= order(), "Order determined by Indices is to small. Order according to the indices " << dimensionCount << ", according to the tensor " << order());
			REQUIRE(dimensionCount <= order(), "Order determined by Indices is to large. Order according to the indices " << dimensionCount << ", according to the tensor " << order());
		}
		
		
		template<class tensor_type>
		std::vector<size_t> IndexedTensorReadOnly<tensor_type>::get_evaluated_dimensions(const std::vector<Index>& _indexOrder) {
			std::vector<size_t> evalDimensions;
			evalDimensions.reserve(_indexOrder.size());
			
			// Find the order that this tensor has using the current indices
			assign_indices();
			size_t trueOrder = 0;
			for(const Index& idx : indices) {
				if(idx.open()) {trueOrder += idx.span; }
			}
			
			// Find dimensions for all given indices
			for(const Index& idx : _indexOrder) {
				if(idx.actual_span(trueOrder) == 0) { continue; }
				
				REQUIRE(misc::count(indices, idx) == 1, "All indices of evaluation target must appear exactly once. Here " << misc::count(indices, idx) << " " << indices << " " << _indexOrder);
				
				// Find index
				size_t indexPos = 0, dimCount = 0;
				while(indices[indexPos] != idx) {
					dimCount += indices[indexPos++].span;
				}
				
				REQUIRE(indices[indexPos].open(), "Index appearing on the LHS of assignment must be open on RHS");
				REQUIRE(dimCount+indices[indexPos].span <= tensorObjectReadOnly->dimensions.size(), "Order determined by Indices is to large. Tensor has " << tensorObjectReadOnly->dimensions.size() << " indices at least " << dimCount+indices[indexPos].span);
				
				// Insert dimensions
				for(size_t i = 0; i < indices[indexPos].span; ++i) {
					evalDimensions.emplace_back(tensorObjectReadOnly->dimensions[dimCount+i]);
				}
			}
			return evalDimensions;
		}
		
		
		// IndexedTensorReadOnly may be instanciated as
		template class IndexedTensorReadOnly<Tensor>;
		template class IndexedTensorReadOnly<TensorNetwork>;
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Aritmetic Operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		template<class tensor_type>
		IndexedTensorMoveable<tensor_type> operator*(const value_t _factor, IndexedTensorReadOnly<tensor_type>&& _tensor) {
			IndexedTensorMoveable<tensor_type> result(std::move(_tensor));
			*result.tensorObject *= _factor;
			return result;
		}
		
		template IndexedTensorMoveable<Tensor> operator*<Tensor>(const value_t _factor, IndexedTensorReadOnly<Tensor>&& _tensor);
		template IndexedTensorMoveable<TensorNetwork> operator*<TensorNetwork>(const value_t _factor, IndexedTensorReadOnly<TensorNetwork>&& _tensor);
		
		
		template<class tensor_type>
		IndexedTensorMoveable<tensor_type> operator*(IndexedTensorReadOnly<tensor_type>&& _tensor, const value_t _factor) {
			return operator*(_factor, std::move(_tensor));
		}
		
		template IndexedTensorMoveable<Tensor> operator*(IndexedTensorReadOnly<Tensor>&& _tensor, const value_t _factor);
		template IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorReadOnly<TensorNetwork>&& _tensor, const value_t _factor);
		
		
		template<class tensor_type>
		IndexedTensorMoveable<tensor_type> operator/(IndexedTensorReadOnly<tensor_type>&& _tensor, const value_t _divisor) {
			IndexedTensorMoveable<tensor_type> result(std::move(_tensor));
			*result.tensorObject /= _divisor;
			return result;
		}
		
		template IndexedTensorMoveable<Tensor> operator/(IndexedTensorReadOnly<Tensor>&& _tensor, const value_t _divisor);
		template IndexedTensorMoveable<TensorNetwork> operator/(IndexedTensorReadOnly<TensorNetwork>&& _tensor, const value_t _divisor);
		
		
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs) {
			IndexedTensorMoveable<Tensor> result(std::move(_lhs));
			result.perform_traces();
			result.indexed_plus_equal(std::move(_rhs));
			return result;
		}
		
		IndexedTensorMoveable<Tensor> operator+(      IndexedTensorMoveable<Tensor> && _lhs, IndexedTensorReadOnly<Tensor>&&  _rhs) {
			IndexedTensorMoveable<Tensor> result(std::move(_lhs));
			result.perform_traces();
			result.indexed_plus_equal(std::move(_rhs));
			return result;
		}
		
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<Tensor>&&  _lhs, IndexedTensorMoveable<Tensor> && _rhs){
			return operator+(std::move(_rhs), std::move(_lhs));
		}
		
		IndexedTensorMoveable<Tensor> operator+(      IndexedTensorMoveable<Tensor> && _lhs, IndexedTensorMoveable<Tensor> && _rhs){
			return operator+(std::move(_lhs), static_cast<IndexedTensorReadOnly<Tensor>&&>(_rhs));
		}
		
		
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs) {
			IndexedTensorMoveable<Tensor> result(std::move(_lhs));
			result.perform_traces();
			result.indexed_minus_equal(std::move(_rhs));
			return result;
		}
		
		IndexedTensorMoveable<Tensor> operator-(      IndexedTensorMoveable<Tensor> && _lhs, IndexedTensorReadOnly<Tensor>&&  _rhs) {
			IndexedTensorMoveable<Tensor> result(std::move(_lhs));
			result.perform_traces();
			result.indexed_minus_equal(std::move(_rhs));
			return result;
		}
		
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<Tensor>&&  _lhs, IndexedTensorMoveable<Tensor> && _rhs){
			return (-1.0)*operator-(std::move(_rhs), std::move(_lhs));
		}
		
		IndexedTensorMoveable<Tensor> operator-(      IndexedTensorMoveable<Tensor> && _lhs, IndexedTensorMoveable<Tensor> && _rhs){
			return operator-(std::move(_lhs), static_cast<IndexedTensorReadOnly<Tensor>&&>(_rhs));
		}
		
		
		
		IndexedTensorMoveable<TensorNetwork> operator+(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			std::unique_ptr<IndexedTensorMoveable<TensorNetwork>> result;
			if(!_lhs.tensorObjectReadOnly->specialized_sum(result, std::move(_lhs), std::move(_rhs)) 
				&& !_rhs.tensorObjectReadOnly->specialized_sum(result, std::move(_rhs), std::move(_lhs))) {
				result.reset( new IndexedTensorMoveable<TensorNetwork>(IndexedTensorMoveable<Tensor>(std::move(_lhs)) + IndexedTensorMoveable<Tensor>(std::move(_rhs))));
				}
				return std::move(*result);
		}
		
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			return std::move(_lhs)+IndexedTensorMoveable<Tensor>(std::move(_rhs));
		}
		
		IndexedTensorMoveable<Tensor> operator+(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs) {
			return IndexedTensorMoveable<Tensor>(std::move(_lhs))+std::move(_rhs);
		}
		
		IndexedTensorMoveable<TensorNetwork> operator-(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			return std::move(_lhs)+(-1*std::move(_rhs));
		}
		
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<Tensor>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			return std::move(_lhs)-IndexedTensorMoveable<Tensor>(std::move(_rhs));
		}
		
		IndexedTensorMoveable<Tensor> operator-(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<Tensor>&& _rhs) {
			return IndexedTensorMoveable<Tensor>(std::move(_lhs))-std::move(_rhs);
		}
		
		
		
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			std::unique_ptr<IndexedTensorMoveable<TensorNetwork>> result(new IndexedTensorMoveable<TensorNetwork>());
			if(!_lhs.tensorObjectReadOnly->specialized_contraction(result, std::move(_lhs), std::move(_rhs)) && !_rhs.tensorObjectReadOnly->specialized_contraction(result, std::move(_rhs), std::move(_lhs))) {
				_lhs.assign_indices();
				result->tensorObject = new TensorNetwork(*_lhs.tensorObjectReadOnly);
				result->tensorObjectReadOnly = result->tensorObject;
				result->indices = _lhs.indices;
				result->deleteTensorObject = true;
				TensorNetwork::add_network_to_network(std::move(*result), std::move(_rhs));
			}
			return std::move(*result);
		}


		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorMoveable<TensorNetwork>&&  _lhs, IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			std::unique_ptr<IndexedTensorMoveable<TensorNetwork>> result(new IndexedTensorMoveable<TensorNetwork>());
			if(!_lhs.tensorObjectReadOnly->specialized_contraction(result, std::move(_lhs), std::move(_rhs)) && !_rhs.tensorObjectReadOnly->specialized_contraction(result, std::move(_rhs), std::move(_lhs))) {
				_lhs.assign_indices();
				result->tensorObject = _lhs.tensorObject;
				result->tensorObjectReadOnly = _lhs.tensorObjectReadOnly;
				result->indices = _lhs.indices;
				result->deleteTensorObject = true;
				_lhs.deleteTensorObject = false;
				TensorNetwork::add_network_to_network(std::move(*result), std::move(_rhs));
			}
			return std::move(*result);
		}
		
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorReadOnly<TensorNetwork>&& _lhs, IndexedTensorMoveable<TensorNetwork>&& _rhs) {
			return operator*(std::move(_rhs), std::move(_lhs));
		}
		
		IndexedTensorMoveable<TensorNetwork> operator*(IndexedTensorMoveable<TensorNetwork>&& _lhs, IndexedTensorMoveable<TensorNetwork>&& _rhs) {
			return operator*(std::move(_lhs), static_cast<IndexedTensorReadOnly<TensorNetwork>&&>(_rhs));
		}
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - External functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		template<class tensor_type>
		value_t frob_norm(const IndexedTensorReadOnly<tensor_type>& _idxTensor) {
			return _idxTensor.tensorObjectReadOnly->frob_norm();
		}
		
		template value_t frob_norm<Tensor>(const IndexedTensorReadOnly<Tensor>& _idxTensor);
		template value_t frob_norm<TensorNetwork>(const IndexedTensorReadOnly<TensorNetwork>& _idxTensor);
		
		value_t one_norm(const IndexedTensorReadOnly<Tensor>& _idxTensor) {
			return _idxTensor.tensorObjectReadOnly->one_norm();
		}
		
		size_t get_eval_order(const std::vector<Index>& _indices) {
			size_t order = 0;
			for(const Index& idx : _indices) {
				INTERNAL_CHECK(idx.flags[Index::Flag::ASSINGED], "Internal Error");
				if(!idx.fixed() && misc::count(_indices, idx) != 2) { order += idx.span; }
			}
			return order;
		}
	} // namespace internal
} // namespace xerus

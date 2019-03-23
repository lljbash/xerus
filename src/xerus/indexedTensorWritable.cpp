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
 * @brief Implementation of the IndexedTensorWritable class.
 */

#include <xerus/misc/containerSupport.h>

#include <xerus/indexedTensorWritable.h>
#include <xerus/indexedTensor.h>
#include <xerus/indexedTensorMoveable.h>

#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/tensorNetwork.h>
#include <xerus/misc/internal.h>

namespace xerus {
	namespace internal {
		template<class tensor_type>
		IndexedTensorWritable<tensor_type>::IndexedTensorWritable(IndexedTensorWritable &&_other ) noexcept : IndexedTensorReadOnly<tensor_type>(std::move(_other)), tensorObject(_other.tensorObject), deleteTensorObject(_other.deleteTensorObject) {
			// Take ownership
			_other.deleteTensorObject = false;
		}
		
		template<class tensor_type>
		IndexedTensorWritable<tensor_type>::IndexedTensorWritable(tensor_type* const _tensorObject, const std::vector<Index>& _indices, const bool _takeOwnership) :
			IndexedTensorReadOnly<tensor_type>(_tensorObject, _indices), tensorObject(_tensorObject), deleteTensorObject(_takeOwnership) {}
			
			
		template<class tensor_type>
		IndexedTensorWritable<tensor_type>::IndexedTensorWritable(tensor_type* const _tensorObject, std::vector<Index>&& _indices, const bool _takeOwnership) :
			IndexedTensorReadOnly<tensor_type>(_tensorObject, std::move(_indices)), tensorObject(_tensorObject), deleteTensorObject(_takeOwnership) {}

		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Destructor - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		template<class tensor_type>
		IndexedTensorWritable<tensor_type>::~IndexedTensorWritable() { 
			if(deleteTensorObject) {
				delete this->tensorObject;
			}
		}
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Other - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		template<class tensor_type>
		bool IndexedTensorWritable<tensor_type>::is_owner() const {
			return deleteTensorObject;
		}
		
		template<>
		void IndexedTensorWritable<Tensor>::indexed_assignement( IndexedTensorReadOnly<Tensor>&& _rhs) {
			if(!_rhs.uses_tensor(tensorObject)) {
				// If LHS and RHS object don't coincide we can directly evaluate
				this->tensorObject->reset(_rhs.get_evaluated_dimensions(indices), Tensor::Initialisation::None);
				evaluate(std::move(*this), std::move(_rhs));
			} else {
				// If the tensors in fact coincide we have to use a tmp object
				IndexedTensorMoveable<Tensor> tmpTensor(std::move(_rhs));
				this->tensorObject->reset(tmpTensor.get_evaluated_dimensions(indices), Tensor::Initialisation::None);
				evaluate(std::move(*this), std::move(tmpTensor));
			}
		}
		
		template<> 
		void IndexedTensorWritable<Tensor>::indexed_assignement(IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			_rhs.tensorObjectReadOnly->require_valid_network();
			IndexedTensorMoveable<TensorNetwork> cpy(std::move(_rhs));
			TensorNetwork::link_traces_and_fix(std::move(cpy));
			
			std::set<size_t> all;
			for (size_t i=0; i < cpy.tensorObject->nodes.size(); ++i) {
				all.insert(i);
			}

			const size_t res = cpy.tensorObject->contract(all);
			
			std::vector<Index> externalOrder;
			for(size_t i = 0; i < cpy.tensorObject->nodes[res].neighbors.size(); ++i) { externalOrder.emplace_back(); }
			
			std::vector<Index> internalOrder;
			for(const TensorNetwork::Link& link: cpy.tensorObject->nodes[res].neighbors) {
				INTERNAL_CHECK(link.external, "Internal Error " << link.other << " " << link.indexPosition);
				internalOrder.emplace_back(externalOrder[link.indexPosition]);
			}
			
			assign_indices(get_eval_order(cpy.indices));
			std::vector<Index> outOrder;
			for (const Index &idx : indices) {
				REQUIRE(misc::contains(cpy.indices, idx), "Every index on the LHS must appear somewhere on the RHS, here: " << cpy.indices << ' ' << indices);
				size_t spanSum = 0;
				for (size_t j = 0; cpy.indices[j] != idx; ++j) {
					INTERNAL_CHECK(j < cpy.indices.size()-1, "ie");
					spanSum += cpy.indices[j].span;
				}
				
				for (size_t i=0; i < idx.span; ++i) {
					outOrder.push_back(externalOrder[spanSum+i]);
				}
			}
		
			(*tensorObject)(outOrder) = (*cpy.tensorObject->nodes[res].tensorObject)(internalOrder);
		}
		
		template<> 
		void IndexedTensorWritable<TensorNetwork>::indexed_assignement(IndexedTensorReadOnly<Tensor>&& _rhs) {
			tensorObject->specialized_evaluation(std::move(*this), IndexedTensorMoveable<TensorNetwork>(new TensorNetwork(*_rhs.tensorObjectReadOnly), _rhs.indices));
		}
		
		template<>
		void IndexedTensorWritable<TensorNetwork>::indexed_assignement(IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			tensorObject->specialized_evaluation(std::move(*this), std::move(_rhs));
		}
		
		template<>
		void IndexedTensorWritable<Tensor>::indexed_plus_equal(IndexedTensorReadOnly<Tensor>&& _rhs) {
			std::unique_ptr<Tensor> reorderedRhs(new Tensor(_rhs.tensorObjectReadOnly->representation));
			(*reorderedRhs)(this->indices) = std::move(_rhs);
			*this->tensorObject += *reorderedRhs;
		}
		
		template<>
		void IndexedTensorWritable<Tensor>::indexed_minus_equal(IndexedTensorReadOnly<Tensor>&& _rhs) {
			std::unique_ptr<Tensor> reorderedRhs(new Tensor(_rhs.tensorObjectReadOnly->representation));
			(*reorderedRhs)(this->indices) = std::move(_rhs);
			
			*this->tensorObject -= *reorderedRhs;
		}
		
		template<>
		void IndexedTensorWritable<TensorNetwork>::indexed_plus_equal(IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			indexed_assignement(std::move(*this) + std::move(_rhs));
		}
		
		template<>
		void IndexedTensorWritable<TensorNetwork>::indexed_minus_equal(IndexedTensorReadOnly<TensorNetwork>&& _rhs) {
			indexed_assignement(std::move(*this) - std::move(_rhs));
		}
		
		template<>
		void IndexedTensorWritable<Tensor>::perform_traces() {
			REQUIRE(deleteTensorObject, "IndexedTensorMoveable must own its tensor object");
			this->assign_indices();
			std::vector<Index> openIndices;
			bool allOpen = true;
			for(const Index& idx : indices) {
				if(idx.open()) {
					openIndices.push_back(idx);
				} else {
					allOpen = false;
				}
			}
			if(!allOpen) { 
				(*this->tensorObject)(openIndices) = std::move(*this);
				this->indices = openIndices;
				indicesAssigned = false;
			}
		}
		
		
		// IndexedTensorReadOnly may be instanciated as
		template class IndexedTensorWritable<Tensor>;
		template class IndexedTensorWritable<TensorNetwork>;
	} // namespace internal
} // namespace xerus

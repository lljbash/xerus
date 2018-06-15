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
 * @brief Implementation of the TTStack classes.
 */

#include <xerus/ttStack.h>
#include <xerus/basic.h>
#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>

#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/ttNetwork.h>
 

namespace xerus {
	namespace internal {
		template<bool isOperator>
		TTStack<isOperator>::TTStack(const bool _canno, const size_t _corePos) : cannonicalization_required(_canno), futureCorePosition(_corePos) {}
		
		
		template<bool isOperator>
		TensorNetwork* TTStack<isOperator>::get_copy() const {
			return new TTStack(*this);
		}
		
		
		template<bool isOperator>
		TTStack<isOperator>::operator TTNetwork<isOperator>() {
			require_valid_network();
			
			if(degree() == 0) {
				std::set<size_t> toContract;
				for(size_t i = 0; i < nodes.size(); ++i) {
					toContract.insert(i);
					contract(toContract);
					return TTNetwork<isOperator>(*nodes[0].tensorObject);
				}
			}
			
			const size_t numComponents = degree()/N;
			const size_t numNodes = degree()/N+2;
			const size_t stackSize = nodes.size()/numNodes;
			
			INTERNAL_CHECK(nodes.size()%numNodes == 0, "IE");
			
			// Contract the stack to a TTNetwork node structure.
			std::set<size_t> toContract;
			for (size_t currentNode = 0; currentNode < numNodes; ++currentNode) {
				toContract.clear();
				for (size_t i = 0; i < stackSize; i++) {
					toContract.insert(currentNode+i*numNodes);
				}
				contract(toContract);
			}
			
			// Reshuffle the nodes to be in the correct order after contraction the nodes will have one of the ids: node, node+numNodes, node+2*numNodes,... (as those were part of the contraction) so modulus gives the correct wanted id.
			reshuffle_nodes([numNodes](const size_t _i){return _i%(numNodes);});
			
			INTERNAL_CHECK(nodes.size() == numNodes, "Internal Error.");
			
			// Reset to new external links
			for(size_t i = 0; i < numComponents; ++i) {
				externalLinks[i].other = i+1;
				externalLinks[i].indexPosition = 1;
			}
			if(isOperator) {
				for(size_t i = 0; i < numComponents; ++i) {
					externalLinks[numComponents+i].other = i+1;
					externalLinks[numComponents+i].indexPosition = 2;
				}
			}
			
			// Fix the first virtual node
			nodes[0].tensorObject->reinterpret_dimensions({1});
			nodes[0].neighbors.resize(1);
			nodes[0].neighbors.front().other = 1;
			nodes[0].neighbors.front().indexPosition = 0;

			// Fix all real components
			std::vector<size_t> shuffle(N+2*stackSize);
			for (size_t i = 1; i+1 < numNodes; ++i) {
				size_t leftCount = 0;
				size_t leftDim = 1, rightDim = 1;
				size_t fullDim = 1;
				for(size_t k = 0; k < N+2*stackSize; ++k) {
					INTERNAL_CHECK(!nodes[i].erased, "IE");
					const TensorNetwork::Link& link = nodes[i].neighbors[k];
					fullDim *= link.dimension;
					if(link.external) {
						if(link.indexPosition < numComponents) {
							shuffle[k] = stackSize;
						} else {
							INTERNAL_CHECK(isOperator, "IE " << link.indexPosition << " vs " << numComponents << " vs " << degree());
							shuffle[k] = stackSize+1;
						}
					} else {
						if(link.other == i-1) {
							shuffle[k] = leftCount++;
							leftDim *= link.dimension;
						} else {
							// We want the order of the next node (so the next one can keep its order)
							size_t otherPos = 0;
							for(const TensorNetwork::Link& otherLink : nodes[i+1].neighbors) {
								if(otherLink.other == i) {
									if(otherLink.indexPosition == k) {
										break;
									} else {
										otherPos++;
									}
								}
							}
							shuffle[k] = stackSize+N+otherPos;
							rightDim *= link.dimension;
						}
					}
				}
				INTERNAL_CHECK(fullDim == nodes[i].tensorObject->size, "Uhh");
				INTERNAL_CHECK(leftCount == stackSize, "IE");
				
				xerus::reshuffle(*nodes[i].tensorObject, *nodes[i].tensorObject, shuffle);
				if(isOperator) {
					nodes[i].tensorObject->reinterpret_dimensions({leftDim, dimensions[i-1], dimensions[i-1+numComponents], rightDim});
				} else {
					nodes[i].tensorObject->reinterpret_dimensions({leftDim, dimensions[i-1], rightDim});
				}
				
				nodes[i].neighbors.clear();
				nodes[i].neighbors.emplace_back(i-1, i==1 ? 0 : N+1, leftDim, false);
				nodes[i].neighbors.emplace_back(0, i-1 , dimensions[i-1], true);
				if(isOperator) { nodes[i].neighbors.emplace_back(0, numComponents+i-1, dimensions[numComponents+i-1], true); }
				nodes[i].neighbors.emplace_back(i+1, 0, rightDim, false);
			}
			
			// Fix the second virtual node
			nodes[numNodes-1].tensorObject->reinterpret_dimensions({1});
			nodes[numNodes-1].neighbors.resize(1);
			nodes[numNodes-1].neighbors.front().other = numNodes-2;
			nodes[numNodes-1].neighbors.front().indexPosition = N+1;
			
			// Create actual TTNetwork
			TTNetwork<isOperator> result;
			static_cast<TensorNetwork&>(result) = static_cast<TensorNetwork&>(*this);
			if(cannonicalization_required) {
				result.canonicalized = false;
				result.move_core(futureCorePosition);
			} else {
				result.canonicalized = true;
				result.corePosition = futureCorePosition;
			}
			result.require_correct_format();
			
			return result;
		}
		
		template<bool isOperator>
		void TTStack<isOperator>::operator*=(const value_t _factor) {
			INTERNAL_CHECK(!nodes.empty(), "There must not be a TTNetwork without any node");
			
			if(cannonicalization_required) {
				*nodes[futureCorePosition+1].tensorObject *= _factor;
			} else if(degree() > 0) {
				*nodes[1].tensorObject *= _factor;
			} else {
				*nodes[0].tensorObject *= _factor;
			}
		}
		
		
		template<bool isOperator>
		void TTStack<isOperator>::operator/=(const value_t _divisor) {
			operator*=(1/_divisor);
		}
		
		
		// TODO get rid of this function and use TTN cast instead
		template<bool isOperator>
		void TTStack<isOperator>::contract_stack(IndexedTensorWritable<TensorNetwork>&& _me) {
			_me.tensorObject->require_valid_network();
			
			if(_me.tensorObject->degree() == 0) {
				std::set<size_t> toContract;
				for(size_t i = 0; i < _me.tensorObject->nodes.size(); ++i) {
					toContract.insert(i);
					_me.tensorObject->contract(toContract);
					_me.tensorObject->reshuffle_nodes([](const size_t _i){return 0;});
					return;
				}
			}
			
			const size_t numComponents = _me.tensorObject->degree()/N;
			const size_t numNodes = _me.tensorObject->degree()/N+2;
			const size_t stackSize = _me.tensorObject->nodes.size()/numNodes;
			
			INTERNAL_CHECK(_me.tensorObject->nodes.size()%numNodes == 0, "IE");
			
			// Contract the stack to a TTNetwork node structure.
			std::set<size_t> toContract;
			for (size_t currentNode = 0; currentNode < numNodes; ++currentNode) {
				toContract.clear();
				for (size_t i = 0; i < stackSize; i++) {
					toContract.insert(currentNode+i*numNodes);
				}
				_me.tensorObject->contract(toContract);
			}
			
			// Reshuffle the nodes to be in the correct order after contraction the nodes will have one of the ids: node, node+numNodes, node+2*numNodes,... (as those were part of the contraction) so modulus gives the correct wanted id.
			_me.tensorObject->reshuffle_nodes([numNodes](const size_t _i){return _i%(numNodes);});
			
			INTERNAL_CHECK(_me.tensorObject->nodes.size() == numNodes, "Internal Error.");
			
			// Reset to new external links
			for(size_t i = 0; i < numComponents; ++i) {
				_me.tensorObject->externalLinks[i].other = i+1;
				_me.tensorObject->externalLinks[i].indexPosition = 1;
			}
			if(isOperator) {
				for(size_t i = 0; i < numComponents; ++i) {
					_me.tensorObject->externalLinks[numComponents+i].other = i+1;
					_me.tensorObject->externalLinks[numComponents+i].indexPosition = 2;
				}
			}
			
			// Fix the first virtual node
			_me.tensorObject->nodes[0].tensorObject->reinterpret_dimensions({1});
			_me.tensorObject->nodes[0].neighbors.resize(1);
			_me.tensorObject->nodes[0].neighbors.front().other = 1;
			_me.tensorObject->nodes[0].neighbors.front().indexPosition = 0;

			// Fix all real components
			std::vector<size_t> shuffle(N+2*stackSize);
			for (size_t i = 1; i+1 < numNodes; ++i) {
				size_t leftCount = 0;
				size_t leftDim = 1, rightDim = 1;
				size_t fullDim = 1;
				for(size_t k = 0; k < N+2*stackSize; ++k) {
					INTERNAL_CHECK(!_me.tensorObject->nodes[i].erased, "IE");
					const TensorNetwork::Link& link = _me.tensorObject->nodes[i].neighbors[k];
					fullDim *= link.dimension;
					if(link.external) {
						if(link.indexPosition < numComponents) {
							shuffle[k] = stackSize;
						} else {
							INTERNAL_CHECK(isOperator, "IE " << link.indexPosition << " vs " << numComponents << " vs " << _me.tensorObject->degree());
							shuffle[k] = stackSize+1;
						}
					} else {
						if(link.other == i-1) {
							shuffle[k] = leftCount++;
							leftDim *= link.dimension;
						} else {
							// We want the order of the next node (so the next one can keep its order)
							size_t otherPos = 0;
							for(const TensorNetwork::Link& otherLink : _me.tensorObject->nodes[i+1].neighbors) {
								if(otherLink.other == i) {
									if(otherLink.indexPosition == k) {
										break;
									} else {
										otherPos++;
									}
								}
							}
							shuffle[k] = stackSize+N+otherPos;
							rightDim *= link.dimension;
						}
					}
				}
				INTERNAL_CHECK(fullDim == _me.tensorObject->nodes[i].tensorObject->size, "Uhh");
				INTERNAL_CHECK(leftCount == stackSize, "IE");
				
				xerus::reshuffle(*_me.tensorObject->nodes[i].tensorObject, *_me.tensorObject->nodes[i].tensorObject, shuffle);
				if(isOperator) {
					_me.tensorObject->nodes[i].tensorObject->reinterpret_dimensions({leftDim, _me.tensorObject->dimensions[i-1], _me.tensorObject->dimensions[i-1+numComponents], rightDim});
				} else {
					_me.tensorObject->nodes[i].tensorObject->reinterpret_dimensions({leftDim, _me.tensorObject->dimensions[i-1], rightDim});
				}
				
				_me.tensorObject->nodes[i].neighbors.clear();
				_me.tensorObject->nodes[i].neighbors.emplace_back(i-1, i==1 ? 0 : N+1, leftDim, false);
				_me.tensorObject->nodes[i].neighbors.emplace_back(0, i-1 , _me.tensorObject->dimensions[i-1], true);
				if(isOperator) { _me.tensorObject->nodes[i].neighbors.emplace_back(0, numComponents+i-1, _me.tensorObject->dimensions[numComponents+i-1], true); }
				_me.tensorObject->nodes[i].neighbors.emplace_back(i+1, 0, rightDim, false);
			}
			
			// Fix the second virtual node
			_me.tensorObject->nodes[numNodes-1].tensorObject->reinterpret_dimensions({1});
			_me.tensorObject->nodes[numNodes-1].neighbors.resize(1);
			_me.tensorObject->nodes[numNodes-1].neighbors.front().other = numNodes-2;
			_me.tensorObject->nodes[numNodes-1].neighbors.front().indexPosition = N+1;
		}
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
		template<bool isOperator>
		void TTStack<isOperator>::specialized_evaluation(IndexedTensorWritable<TensorNetwork>&&   /*_me*/, IndexedTensorReadOnly<TensorNetwork>&&  /*_other*/) {
			LOG(fatal, "TTStack not supported as a storing type");
		}
		
		
		template<bool isOperator>
		bool TTStack<isOperator>::specialized_contraction(std::unique_ptr<IndexedTensorMoveable<TensorNetwork>>& _out, IndexedTensorReadOnly<TensorNetwork>&& _me, IndexedTensorReadOnly<TensorNetwork>&& _other) const {
			return TTNetwork<isOperator>::specialized_contraction_f(_out, std::move(_me), std::move(_other));
		}
		
		
		template<bool isOperator>
		bool TTStack<isOperator>::specialized_sum(std::unique_ptr<IndexedTensorMoveable<TensorNetwork>>& _out, IndexedTensorReadOnly<TensorNetwork>&& _me, IndexedTensorReadOnly<TensorNetwork>&& _other) const {
			return TTNetwork<isOperator>::specialized_sum_f(_out, std::move(_me), std::move(_other));
		}
		
		
		template<bool isOperator>
		value_t TTStack<isOperator>::frob_norm() const {
			const Index i;
			TTNetwork<isOperator> tmp;
			tmp(i&0) = IndexedTensorMoveable<TensorNetwork>(this->get_copy(), {i&0});
			return tmp.frob_norm();
		}
		
		
		// Explicit instantiation of the two template parameters that will be implemented in the xerus library
		template class TTStack<false>;
		template class TTStack<true>;
	} // namespace internal
} // namespace xerus

// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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
* @brief Implementation of the TTNetwork class (and thus TTTensor and TTOperator).
*/

#include <algorithm>
#include <memory>

#include <xerus/ttNetwork.h>

#include <xerus/misc/check.h>
#include <xerus/misc/math.h>
#include <xerus/misc/performanceAnalysis.h>
#include <xerus/misc/internal.h>

#include <xerus/basic.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/ttStack.h>
#include <xerus/indexedTensorList.h>
#include <xerus/indexedTensorMoveable.h>

namespace xerus {
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork() : TensorNetwork(), canonicalized(false) {}
	
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const Tensor& _tensor, const double _eps, const size_t _maxRank) :
		TTNetwork(_tensor, _eps, std::vector<size_t>(_tensor.degree() == 0 ? 0 : _tensor.degree()/N-1, _maxRank)) {}
	
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const size_t _degree) : TTNetwork(std::vector<size_t>(_degree, 1)) { }
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(Tensor::DimensionTuple _dimensions) : TensorNetwork(ZeroNode::None), canonicalized(true), corePosition(0) {
		dimensions = std::move(_dimensions);
		
		REQUIRE(dimensions.size()%N==0, "Illegal degree for TTOperator.");
		REQUIRE(!misc::contains(dimensions, size_t(0)), "Zero is no valid dimension.");
		const size_t numComponents = dimensions.size()/N;
		
		if (numComponents == 0) {
			nodes.emplace_back(std::make_unique<Tensor>());
			return;
		}
		
		// ExternalLinks
		externalLinks.reserve(dimensions.size());
		for (size_t i = 0; i < numComponents; ++i) {
			externalLinks.emplace_back(i+1, 1, dimensions[i], false);
		}
		
		if (isOperator) {
			for (size_t i = 0; i < numComponents; ++i) {
				externalLinks.emplace_back(i+1, 2, dimensions[numComponents+i], false);
			}
		}
		
		std::vector<TensorNetwork::Link> neighbors;
		
		neighbors.emplace_back(1, 0, 1, false);
		
		nodes.emplace_back(std::make_unique<Tensor>(Tensor::ones({1})), std::move(neighbors));
		
		for (size_t i = 0; i < numComponents; ++i) {
			neighbors.clear();
			neighbors.emplace_back(i, i==0 ? 0 : N+1, 1, false);
			neighbors.emplace_back(-1, i, dimensions[i], true);
			if(isOperator) { neighbors.emplace_back(-1, i+numComponents, dimensions[numComponents+i], true); }
			neighbors.emplace_back(i+2, 0, 1, false);
			
			if(!isOperator) {
				nodes.emplace_back( std::make_unique<Tensor>(Tensor::dirac({1, dimensions[i], 1}, 0)), std::move(neighbors) );
			} else {
				nodes.emplace_back( std::make_unique<Tensor>(Tensor::dirac({1, dimensions[i], dimensions[numComponents+i], 1}, 0)), std::move(neighbors) );
			}
		}
		
		neighbors.clear();
		neighbors.emplace_back(numComponents, N+1, 1, false);
		nodes.emplace_back( std::make_unique<Tensor>(Tensor::ones({1})), std::move(neighbors));
		
		// Make a Zero Tensor (at core)
		(*nodes[1].tensorObject)[0] = 0;
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const Tensor& _tensor, const double _eps, const TensorNetwork::RankTuple& _maxRanks): TTNetwork(_tensor.degree()) {
		REQUIRE(_tensor.degree()%N==0, "Number of indicis must be even for TTOperator");
		REQUIRE(_eps >= 0 && _eps < 1, "_eps must be positive and smaller than one. " << _eps << " was given.");
		REQUIRE(_maxRanks.size() == num_ranks(), "We need " << num_ranks() <<" ranks but " << _maxRanks.size() << " where given");
		REQUIRE(!misc::contains(_maxRanks, size_t(0)), "Maximal ranks must be strictly positive. Here: " << _maxRanks);
		
		const size_t numComponents = degree()/N;
		
		if (_tensor.degree() == 0) {
			*nodes[0].tensorObject = _tensor;
			return;
		}
		
		dimensions = _tensor.dimensions;
		
		Tensor remains;
		if(isOperator) {
			std::vector<size_t> shuffle(_tensor.degree());
			for(size_t i = 0; i < numComponents; ++i) {
				shuffle[i] = 2*i;
				shuffle[numComponents + i] = 2*i+1; 
			}
			
			xerus::reshuffle(remains, _tensor, shuffle);
		} else {
			remains = _tensor;
		}
		
		// Add ghost dimensions used in the nodes
		std::vector<size_t> extDimensions;
		extDimensions.reserve(remains.degree()+2);
		extDimensions.emplace_back(1);
		extDimensions.insert(extDimensions.end(), remains.dimensions.begin(), remains.dimensions.end());
		extDimensions.emplace_back(1);
		remains.reinterpret_dimensions(extDimensions);
		
		
		Tensor singularValues, newNode;
        auto eps = _eps / std::sqrt(_tensor.degree() - 1);
		for(size_t position = numComponents-1; position > 0; --position) {
			calculate_svd(remains, singularValues, newNode, remains, 1+position*N, _maxRanks[position-1], eps);
			
			set_component(position, std::move(newNode)); 
			newNode.reset();
			xerus::contract(remains, remains, false, singularValues, false, 1);
		}
		
		set_component(0, remains);
		assume_core_position(0);
	}
	

// 	template<bool isOperator>
// 	TTNetwork<isOperator>::TTNetwork(const TensorNetwork &_network, double _eps) : TTNetwork(Tensor(_network)) {
// 		LOG(warning, "Cast of arbitrary tensor network to TT not yet supported. Casting to Tensor first"); // TODO
// 	}

	
	template<bool isOperator>
	TTNetwork<isOperator> TTNetwork<isOperator>::ones(const std::vector<size_t>& _dimensions) {
		REQUIRE(_dimensions.size()%N == 0, "Illegal number of dimensions for ttOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a TTTensor with dimension 0 is not possible.");
		
		if(_dimensions.empty()) {
			return TTNetwork(Tensor::ones({}));
		}
		
		TTNetwork result(_dimensions.size());
		const size_t numNodes = _dimensions.size()/N;
		
		std::vector<size_t> dimensions(isOperator ? 4 : 3, 1);
		for(size_t i = 0; i < numNodes; ++i) {
			dimensions[1] = _dimensions[i];
			if (isOperator) {
				dimensions[2] = _dimensions[i+numNodes];
			}
			result.set_component(i, Tensor::ones(dimensions));
		}
		result.canonicalize_left();
		return result;
	}
	
	
	template<> template<>
	TTNetwork<true> TTNetwork<true>::identity(const std::vector<size_t>& _dimensions) {
		REQUIRE(_dimensions.size()%N==0, "Illegal number of dimensions for ttOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a TTTensor with dimension 0 is not possible.");
		
		if(_dimensions.empty()) {
			return TTNetwork(Tensor::ones({}));
		}
		
		const size_t numComponents = _dimensions.size()/N;
		
		TTNetwork result(_dimensions.size());
		
		std::vector<size_t> constructionVector(4, 1);
		for (size_t i = 0; i < numComponents; ++i) {
			constructionVector[1] = _dimensions[i];
			constructionVector[2] = _dimensions[i+numComponents];
			result.set_component(i, Tensor(constructionVector, [](const std::vector<size_t> &_idx){
				if (_idx[1] == _idx[2]) {
					return 1.0;
				}
				return 0.0;
			}));
		}
		
		result.canonicalize_left();
		return result;
	}
	
	template<bool isOperator>
	TTNetwork<isOperator> TTNetwork<isOperator>::kronecker(const std::vector<size_t>& _dimensions) {
		REQUIRE(_dimensions.size()%N == 0, "Illegal number of dimensions for ttOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a TTNetwork with dimension 0 is not possible.");
		
		if(_dimensions.empty()) {
			return TTNetwork(Tensor::kronecker({}));
		}
		
		TTNetwork result(_dimensions.size());
		const size_t numNodes = _dimensions.size()/N;
		
		const auto minN = misc::min(_dimensions);
		
		// All nodes are simply kronecker tensors themself
		std::vector<size_t> dimensions;
		for(size_t i = 0; i < numNodes; ++i) {
			dimensions.reserve(4);
			if(i > 0) { dimensions.push_back(minN); }
			dimensions.push_back(_dimensions[i]);
			if (isOperator) { dimensions.push_back(_dimensions[i+numNodes]); }
			if(i+1 < numNodes) { dimensions.push_back(minN); }
			auto newComp = Tensor::kronecker(dimensions);
			if(i == 0) { dimensions.insert(dimensions.begin(), 1); }
			if(i+1 == numNodes) { dimensions.insert(dimensions.end(), 1); }
			if(i == 0 || i+1 == numNodes) { newComp.reinterpret_dimensions(std::move(dimensions)); }
			result.set_component(i, std::move(newComp));
			dimensions.clear();
		}
		result.canonicalize_left();
		return result;
	}
	
	template<bool isOperator>
	TTNetwork<isOperator> TTNetwork<isOperator>::dirac(std::vector<size_t> _dimensions, const std::vector<size_t>& _position) {
		REQUIRE(_dimensions.size()%N==0, "Illegal number of dimensions for ttOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a TTTensor with dimension 0 is not possible.");
		REQUIRE(_dimensions.size() == _position.size(), "Inconsitend number of entries in _dimensions and _position.");
		
		const size_t numComponents = _dimensions.size()/N;
		
		if(numComponents <= 1) {
			return TTNetwork(Tensor::dirac(_dimensions, _position));
		}
		
		TTNetwork result(_dimensions);
		
		for (size_t i = 0; i < numComponents; ++i) {
			if(isOperator) {
				result.set_component(i, Tensor::dirac({1, result.dimensions[i], result.dimensions[numComponents+i], 1}, _position[i]*result.dimensions[numComponents+i] + _position[numComponents+i]));
			} else {
				result.set_component(i, Tensor::dirac({1, result.dimensions[i], 1}, _position[i]));
			}
		}
		return result;
	}
	
	template<bool isOperator>
	TTNetwork<isOperator> TTNetwork<isOperator>::dirac(std::vector<size_t> _dimensions, const size_t _position) {
		return dirac(_dimensions, Tensor::position_to_multiIndex(_position, _dimensions));
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	
	#ifndef XERUS_DISABLE_RUNTIME_CHECKS
		template<bool isOperator>
		void TTNetwork<isOperator>::require_correct_format() const {
			require_valid_network(); // Network must at least be valid.
			
			const size_t numComponents = degree()/N;
			const size_t numNodes = degree() == 0 ? 1 : degree()/N + 2;
			REQUIRE(nodes.size() == numNodes, "Wrong number of nodes: " << nodes.size() << " expected " << numNodes << ".");
			REQUIRE(!canonicalized || (degree() == 0 && corePosition == 0) || corePosition < numComponents, "Invalid corePosition: " << corePosition << " there are only " << numComponents << " components.");
			
			// Per external link
			for (size_t n = 0; n < externalLinks.size(); ++n) {
				const TensorNetwork::Link &l = externalLinks[n];
				REQUIRE(l.other == (n%numComponents)+1, "The " << n << "-th external link must point the the " << (n%numComponents) << "-th component (i.e. the " << (n%numComponents)+1 << "-th node) but does point to the " << l.other << "-th node.");
			}
			
			// Virtual nodes
			if(degree() > 0) {
				REQUIRE(nodes.front().degree() == 1, "The left virtual node must have degree 1, but has size " << nodes.front().degree());
				REQUIRE(nodes.front().neighbors[0].dimension == 1, "The left virtual node's single dimension must be 1, but is " << nodes.front().neighbors[0].dimension);
				REQUIRE(nodes.front().neighbors[0].other == 1, "The left virtual node's single link must be to node 1, but is towards node " << nodes.front().neighbors[0].other);
				REQUIRE(nodes.front().neighbors[0].indexPosition == 0, "The left virtual node's single link must link at indexPosition 0, but link at " << nodes.front().neighbors[0].indexPosition);
				REQUIRE(misc::hard_equal((*nodes.front().tensorObject)[0], 1.0), "The left virtual node's single entry must be 1.0, but it is " << (*nodes.front().tensorObject)[0]);
				REQUIRE(!nodes.front().tensorObject->has_factor(), "The left virtual node must no carry a non-trivial factor.");
				
				REQUIRE(nodes.back().degree() == 1, "The right virtual node must have degree 1, but has size " << nodes.back().degree());
				REQUIRE(nodes.back().neighbors[0].dimension == 1, "The right virtual node's single dimension must be 1, but is " << nodes.back().neighbors[0].dimension);
				REQUIRE(nodes.back().neighbors[0].other == numNodes-2, "The right virtual node's single link must be to node " << numNodes-2 << ", but is towards node " << nodes.back().neighbors[0].other);
				REQUIRE(nodes.back().neighbors[0].indexPosition == N+1, "The right virtual node's single link must link at indexPosition " << N+1 << ", but link at " << nodes.back().neighbors[0].indexPosition);
				REQUIRE(misc::hard_equal((*nodes.back().tensorObject)[0], 1.0), "The right virtual node's single entry must be 1.0, but it is " << (*nodes.back().tensorObject)[0]);
				REQUIRE(!nodes.back().tensorObject->has_factor(), "The right virtual node must no carry a non-trivial factor.");
			}
			
			// Per component
			for (size_t n = 0; n < numComponents; ++n) {
				const TensorNode& node = nodes[n+1];
				
				REQUIRE(!canonicalized || n == corePosition || !node.tensorObject->has_factor(), "In canonicalized TTNetworks only the core may carry a non-trivial factor. Violated by component " << n << " factor: " << node.tensorObject->factor << " corepos: " << corePosition);
				
				REQUIRE(node.degree() == N+2, "Every TT-Component must have degree " << N+2 << ", but component " << n << " has degree " << node.degree());
				REQUIRE(!node.neighbors[0].external, "The first link of each TT-Component must not be external. Violated by component " << n);
				REQUIRE(node.neighbors[0].other == n, "The first link of each TT-Component must link to the previous node. Violated by component " << n << ", which instead links to node " << node.neighbors[0].other << " (expected " << n << ").");
				REQUIRE(node.neighbors[0].indexPosition == (n==0?0:N+1), "The first link of each TT-Component must link to the last last index of the previous node. Violated by component " << n << ", which instead links to index " << node.neighbors[0].indexPosition << " (expected " << (n==0?0:N+1) << ").");
				
				REQUIRE(node.neighbors[1].external, "The second link of each TT-Component must be external. Violated by component " << n << ".");
				REQUIRE(node.neighbors[1].indexPosition == n, "The second link of each TT-Component must link to the external dimension equal to the component position. Violated by component " << n << " which links at " << node.neighbors[1].indexPosition);
				REQUIRE(!isOperator || node.neighbors[2].external, "The third link of each TTO-Component must be external. Violated by component " << n << ".");
				REQUIRE(!isOperator || node.neighbors[2].indexPosition == numComponents+n, "The third link of each TTO-Component must link to the external dimension equal to the component position + numComponents. Violated by component " << n << " which links at " << node.neighbors[2].indexPosition << " (expected " << numComponents+n << ").");
				
				REQUIRE(!node.neighbors.back().external, "The last link of each TT-Component must not be external. Violated by component " << n);
				REQUIRE(node.neighbors.back().other == n+2, "The last link of each TT-Component must link to the next node. Violated by component " << n << ", which instead links to node " << node.neighbors.back().other << " (expected " << n+2 << ").");
				REQUIRE(node.neighbors.back().indexPosition == 0, "The last link of each TT-Component must link to the first index of the next node. Violated by component " << n << ", which instead links to index " << node.neighbors.back().indexPosition << " (expected 0).");
			}
		}
	#else
		template<bool isOperator>
		void TTNetwork<isOperator>::require_correct_format() const { }
	#endif
	
	
	template<bool isOperator>
	bool TTNetwork<isOperator>::exceeds_maximal_ranks() const {
		const size_t numComponents = dimensions.size()/N;
		for (size_t i = 0; i < numComponents; ++i) {
			const Tensor& comp = get_component(i);
			const size_t extDim = isOperator ? comp.dimensions[1]*comp.dimensions[2] : comp.dimensions[1];
			if (comp.dimensions.front() > extDim * comp.dimensions.back() || comp.dimensions.back() > extDim * comp.dimensions.front()) {
				return true;
			}
		}
		return false;
	}
	
	
	template<bool isOperator>
	size_t TTNetwork<isOperator>::num_ranks() const {
		return degree() == 0 ? 0 : degree()/N-1;
	}
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	
	template<bool isOperator>
	std::vector<size_t> TTNetwork<isOperator>::reduce_to_maximal_ranks(std::vector<size_t> _ranks, const std::vector<size_t>& _dimensions) {
		const size_t numComponents = _dimensions.size()/N;
		REQUIRE(_dimensions.size()%N == 0, "invalid number of dimensions for TTOperator");
		REQUIRE(numComponents == _ranks.size()+1, "Invalid number of ranks ("<<_ranks.size()<<") or dimensions ("<<_dimensions.size()<<") given.");
		
		// Left to right sweep
		size_t currMax = 1;
		for (size_t i = 0; i+1 < numComponents; ++i) {
			currMax *= _dimensions[i];
			if (isOperator) { currMax *= _dimensions[numComponents+i]; }
			
			if (currMax < _ranks[i]) { 
				_ranks[i] = currMax;
			} else {
				currMax = _ranks[i];
			}
		}
	
		// Right to left sweep
		currMax = 1;
		for (size_t i = 1; i < numComponents; ++i) {
			currMax *= _dimensions[numComponents-i];
			if (isOperator) { currMax *= _dimensions[2*numComponents-i]; }
			
			if (currMax < _ranks[numComponents-i-1]) {
				_ranks[numComponents-i-1] = currMax;
			} else {
				currMax = _ranks[numComponents-i-1];
			}
		}
		
		return _ranks;
	}
	
	
	template<bool isOperator>
	size_t TTNetwork<isOperator>::degrees_of_freedom(const std::vector<size_t> &_dimensions, const std::vector<size_t> &_ranks) {
		if (_dimensions.empty()) { return 1; }
		const size_t numComponents = _dimensions.size()/N;
		REQUIRE(_dimensions.size()%N == 0, "invalid number of dimensions for TTOperator");
		REQUIRE(numComponents == _ranks.size()+1, "Invalid number of ranks ("<<_ranks.size()<<") or dimensions ("<<_dimensions.size()<<") given.");
		size_t result = 0;
		for (size_t i=0; i<numComponents; ++i) {
			size_t component = i==0? 1 : _ranks[i-1];
			component *= _dimensions[i];
			if (isOperator) { component *= _dimensions[i+numComponents]; }
			if (i<_ranks.size()) { component *= _ranks[i]; }
			result += component;
		}
		for (const auto r : _ranks) {
			result -= misc::sqr(r);
		}
		return result;
	}
	
	template<bool isOperator>
	size_t TTNetwork<isOperator>::degrees_of_freedom() {
		return degrees_of_freedom(dimensions, ranks());
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::fix_mode(const size_t _mode, const size_t _slatePosition) {
		REQUIRE(!isOperator, "fix_mode(), does not work for TTOperators, if applicable cast to TensorNetwork first");
		TensorNetwork::fix_mode(_mode, _slatePosition);
	}
	
	template<bool isOperator>
	void TTNetwork<isOperator>::resize_mode(const size_t _mode, const size_t _newDim, const size_t _cutPos) {
		TensorNetwork::resize_mode(_mode, _newDim, _cutPos);
		if(canonicalized && _newDim != corePosition) {
			const size_t oldCorePosition = corePosition;
			const size_t numComponents = degree()/N;
			move_core(_mode%numComponents);
			move_core(oldCorePosition);
		}
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::use_dense_representations() {
		for (size_t i = 0; i < degree(); ++i) { 
			component(i).use_dense_representation(); 
		}
	}
	
	template<bool isOperator>
	Tensor& TTNetwork<isOperator>::component(const size_t _idx) {
		REQUIRE(_idx == 0 || _idx < degree()/N, "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree()/N << " components.");
		return *nodes[degree() == 0 ? 0 : _idx+1].tensorObject;
	}
	
	
	template<bool isOperator>
	const Tensor& TTNetwork<isOperator>::get_component(const size_t _idx) const {
		REQUIRE(_idx == 0 || _idx < degree()/N, "Illegal index " << _idx <<" in TTNetwork::get_component.");
		return *nodes[degree() == 0 ? 0 : _idx+1].tensorObject;
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::set_component(const size_t _idx, Tensor _T) {
		if(degree() == 0) {
			REQUIRE(_idx == 0, "Illegal index " << _idx <<" in TTNetwork::set_component");
			REQUIRE(_T.degree() == 0, "Component of degree zero TTNetwork must have degree zero. Given: " << _T.degree());
			*nodes[0].tensorObject = std::move(_T);
		} else {
			REQUIRE(_idx < degree()/N, "Illegal index " << _idx <<" in TTNetwork::set_component");
            REQUIRE(_T.degree() == N+2, "Component " << _idx << " must have degree " << N+2 << ". Given: " << _T.degree());
			
			TensorNode& currNode = nodes[_idx+1];
			*currNode.tensorObject = std::move(_T);
			for (size_t i = 0; i < N+2; ++i) {
				currNode.neighbors[i].dimension = currNode.tensorObject->dimensions[i];
				if (currNode.neighbors[i].external) {
					externalLinks[currNode.neighbors[i].indexPosition].dimension = currNode.tensorObject->dimensions[i];
					dimensions[currNode.neighbors[i].indexPosition] = currNode.tensorObject->dimensions[i];
				}
			}
		}
		
		canonicalized = canonicalized && (corePosition == _idx);
	}
	
	
	template<bool isOperator>
	std::vector<std::vector<std::tuple<size_t, size_t, value_t>>> get_grouped_entries(const Tensor& _component) {
		REQUIRE(_component.is_sparse(), "Not usefull (and not implemented) for dense Tensors.");
		
		const size_t externalDim = isOperator ? _component.dimensions[1] * _component.dimensions[2] : _component.dimensions[1];
		
		std::vector<std::vector<std::tuple<size_t, size_t, value_t>>> groups(externalDim);
		
		for(const auto& entry : _component.get_unsanitized_sparse_data()) {
			const size_t r2 = entry.first%_component.dimensions.back();
			const size_t n = (entry.first/_component.dimensions.back())%externalDim;
			const size_t r1 = (entry.first/_component.dimensions.back())/externalDim;
			groups[n].emplace_back(r1, r2, _component.factor*entry.second);
		}
		
		return groups;
	}
	
	
	template<bool isOperator>
	std::pair<TensorNetwork, TensorNetwork> TTNetwork<isOperator>::chop(const size_t _position) const {
		require_correct_format();
		
		const size_t numComponents = degree()/N;
		REQUIRE(_position < numComponents, "Can't split a " << numComponents << " component TTNetwork at position " << _position);
		
		// Create the resulting TNs
		TensorNetwork left(ZeroNode::None);
		TensorNetwork right(ZeroNode::None);
		
		left.nodes.push_back(nodes[0]);
		for (size_t i = 0; i < _position; ++i) {
			left.dimensions.push_back(dimensions[i]);
			left.externalLinks.push_back(externalLinks[i]);
			left.nodes.push_back(nodes[i+1]);
		}
		if(isOperator) {
			for(size_t i = 0; i < _position; ++i) {
				left.dimensions.push_back(dimensions[i+numComponents]);
				left.externalLinks.push_back(externalLinks[i+numComponents]);
			}
		}
		left.dimensions.push_back(left.nodes.back().neighbors.back().dimension);
		left.externalLinks.emplace_back(_position, _position==0?0:N+1, left.nodes.back().neighbors.back().dimension , false);
		left.nodes.back().neighbors.back().external = true;
		left.nodes.back().neighbors.back().indexPosition = isOperator ? 2*_position-1 : _position;
		
		right.dimensions.push_back(nodes[_position+2].neighbors.front().dimension);
		right.externalLinks.emplace_back(_position+2, 0, nodes[_position+2].neighbors.front().dimension , false); // NOTE other will be corrected to 0 in the following steps

		for(size_t i = _position+1; i < numComponents; ++i) {
			right.dimensions.push_back(dimensions[i]);
			right.externalLinks.push_back(externalLinks[i]);
			right.nodes.push_back(nodes[i+1]);
		}
		if(isOperator) {
			for(size_t i = _position+1; i < numComponents+1; ++i) {
				right.dimensions.push_back(dimensions[i+numComponents]);
				right.externalLinks.push_back(externalLinks[i+numComponents]);
			}
		}
		// The last node
		right.nodes.push_back(nodes.back());
		
		right.nodes.front().neighbors.front().external = true;
		right.nodes.front().neighbors.front().indexPosition = _position; // NOTE indexPosition will be corrected to 0 in the following steps
		
		// Account for the fact that the first _position+2 nodes do not exist
		for(TensorNetwork::Link& link : right.externalLinks) {
			link.other -= _position+2;
		}
		
		for(TensorNode& node : right.nodes) {
			for(TensorNetwork::Link& link : node.neighbors) {
				if(link.external) {
					link.indexPosition -= _position;
				} else {
					link.other -= _position+2;
				}
			}
		}
		
		return std::pair<TensorNetwork, TensorNetwork>(std::move(left), std::move(right));
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::move_core(const size_t _position, const bool _keepRank) {
		const size_t numComponents = degree()/N;
		REQUIRE(_position < numComponents || (_position == 0 && degree() == 0), "Illegal core-position " << _position << " chosen for TTNetwork with " << numComponents << " components");
		require_correct_format();
		
		if(canonicalized) {
			// Move right?
			for (size_t n = corePosition; n < _position; ++n) {
				transfer_core(n+1, n+2, !_keepRank);
			}
			
			// Move left?
			for (size_t n = corePosition; n > _position; --n) {
				transfer_core(n+1, n, !_keepRank);
			}
		} else {
			// Move right?
			for (size_t n = 0; n < _position; ++n) {
				transfer_core(n+1, n+2, !_keepRank);
			}
			
			// Move left?
			for (size_t n = numComponents; n > _position+1; --n) {
				transfer_core(n, n-1, !_keepRank);
			}
		}
		
		while (exceeds_maximal_ranks()) {
			// Move left from given CorePosition
			for (size_t n = _position; n > 0; --n) {
				transfer_core(n+1, n, !_keepRank);
			}
			
			// Move to the most right
			for (size_t n = 0; n+1 < numComponents; ++n) {
				transfer_core(n+1, n+2, !_keepRank);
			}
			
			// Move back left to given CorePosition
			for (size_t n = numComponents; n > _position+1; --n) {
				transfer_core(n, n-1, !_keepRank);
			}
		}
		
		canonicalized = true;
		corePosition = _position;
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::canonicalize_left() {
		move_core(0);
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::canonicalize_right() {
		move_core(degree() == 0 ? 0 : degree()/N-1);
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::round(const std::vector<size_t>& _maxRanks, const double _eps) {
		require_correct_format();
		const size_t numComponents = degree()/N;
		REQUIRE(_eps < 1, "_eps must be smaller than one. " << _eps << " was given.");
		REQUIRE(_maxRanks.size()+1 == numComponents || (_maxRanks.empty() && numComponents == 0), "There must be exactly degree/N-1 maxRanks. Here " << _maxRanks.size() << " instead of " << numComponents-1 << " are given.");
		REQUIRE(!misc::contains(_maxRanks, size_t(0)), "Trying to round a TTTensor to rank 0 is not possible.");
		
		const bool initialCanonicalization = canonicalized;
		const size_t initialCorePosition = corePosition;
		
		canonicalize_right();
		
		for(size_t i = 0; i+1 < numComponents; ++i) {
			round_edge(numComponents-i, numComponents-i-1, _maxRanks[numComponents-i-2], _eps, 0.0);
		}
		
		assume_core_position(0);
		
		if(initialCanonicalization) {
			move_core(initialCorePosition);
		}
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::round(const size_t _maxRank) {
		round(std::vector<size_t>(num_ranks(), _maxRank), EPSILON);
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::round(const int _maxRank) {
		REQUIRE( _maxRank > 0, "MaxRank must be positive");
		round(size_t(_maxRank));
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::round(const value_t _eps) {
		round(std::vector<size_t>(num_ranks(), std::numeric_limits<size_t>::max()), _eps);
	}

	
	template<bool isOperator>
	void TTNetwork<isOperator>::soft_threshold(const std::vector<double> &_taus, const bool  /*_preventZero*/) {
		const size_t numComponents = degree()/N;
		REQUIRE(_taus.size()+1 == numComponents || (_taus.empty() && numComponents == 0), "There must be exactly degree/N-1 taus. Here " << _taus.size() << " instead of " << numComponents-1 << " are given.");
		require_correct_format();
		
		const bool initialCanonicalization = canonicalized;
		const size_t initialCorePosition = corePosition;
		
		canonicalize_right();
		
		for(size_t i = 0; i+1 < numComponents; ++i) {
			round_edge(numComponents-i, numComponents-i-1, std::numeric_limits<size_t>::max(), 0.0, _taus[i]);
		}
		
		assume_core_position(0);
		
		if(initialCanonicalization) {
			move_core(initialCorePosition);
		}
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::soft_threshold(const double _tau, const bool _preventZero) {
		soft_threshold(std::vector<double>(num_ranks(), _tau), _preventZero);
	}
	

	template<bool isOperator>
	std::vector<size_t> TTNetwork<isOperator>::ranks() const {
		std::vector<size_t> res;
		res.reserve(num_ranks());
		for (size_t n = 1; n+2 < nodes.size(); ++n) {
			res.push_back(nodes[n].neighbors.back().dimension);
		}
		return res;
	}
	
	
	template<bool isOperator>
	size_t TTNetwork<isOperator>::rank(const size_t _i) const {
		REQUIRE(_i+1 < degree()/N, "Requested illegal rank " << _i);
		return nodes[_i+1].neighbors.back().dimension;
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::assume_core_position(const size_t _pos) {
		REQUIRE(_pos < degree()/N || (degree() == 0 && _pos == 0), "Invalid core position.");
		corePosition = _pos;
		canonicalized = true;
	}
	
	
	template<bool isOperator>
	TensorNetwork* TTNetwork<isOperator>::get_copy() const {
		return new TTNetwork(*this);
	}
	
	template<bool isOperator>
	void TTNetwork<isOperator>::contract_unconnected_subnetworks() {
		if(degree() == 0) {
			std::set<size_t> all;
			for(size_t i = 0; i < nodes.size(); ++i) { all.emplace_hint(all.end(), i); }
			contract(all);
			canonicalized = false;
		} else {
			REQUIRE(nodes.size() > 2, "Invalid TTNetwork");
			const size_t numComponents = nodes.size()-2;
			
			for(size_t i = 0; i+1 < numComponents; ++i) {
				if(nodes[i+1].degree() == 2) {
						// If we are the core, everything is fine, we contract ourself to the next node, then get removed and the corePositions stays. If the next Node is the core, we have to change the corePosition to ours, because we will be removed. In all other cases cannonicalization is destroyed.
						if(corePosition == i+1) { corePosition = i; }
						else if(corePosition != i) { canonicalized = false; }
						contract(i+1, i+2);
				}
			}
			
			// Extra treatment for last component to avoid contraction to the pseudo-node.
			if(nodes[numComponents].degree() == 2) {
				if(corePosition == numComponents-1) { corePosition = numComponents-2; }
				else if(corePosition != numComponents-2) { canonicalized = false; }
				contract(numComponents-1, numComponents);
			}
		}
		
		INTERNAL_CHECK(corePosition < degree() || !canonicalized, "Woot");
		
		sanitize();
	}
	
	
	template<bool isOperator>
	value_t TTNetwork<isOperator>::frob_norm() const {
		require_correct_format();
		if (canonicalized) {
			return get_component(corePosition).frob_norm();
		}
		const Index i;
		return std::sqrt(value_t((*this)(i&0)*(*this)(i&0)));
	}
	
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - -  Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - */
	
    // TODO why sparse?
	template<bool isOperator>
	TTNetwork<isOperator>& TTNetwork<isOperator>::operator+=(const TTNetwork<isOperator>& _other) {
		REQUIRE(dimensions == _other.dimensions, "The dimensions in TT sum must coincide. Given " << dimensions << " vs " << _other.dimensions);
		require_correct_format();
		
		const size_t numComponents = degree()/N;
		
		const bool initialCanonicalization = canonicalized;
		const size_t initialCorePosition = corePosition;
		
		if (numComponents <= 1) {
			component(0) += _other.get_component(0);
			return *this;
		}
		
		XERUS_PA_START;
		for(size_t position = 0; position < numComponents; ++position) {
			// Get current components
			const Tensor& myComponent = get_component(position);
			const Tensor& otherComponent = _other.get_component(position);
			
			// Structure has to be (for degree 4)
			// (L1 R1) * ( L2 0  ) * ( L3 0  ) * ( L4 )
			// 			 ( 0  R2 )   ( 0  R3 )   ( R4 )
			
			// Create a Tensor for the result
			std::vector<size_t> nxtDimensions;
			nxtDimensions.emplace_back(position == 0 ? 1 : myComponent.dimensions.front()+otherComponent.dimensions.front());
			nxtDimensions.emplace_back(myComponent.dimensions[1]);
			if (isOperator) { nxtDimensions.emplace_back(myComponent.dimensions[2]); }
			nxtDimensions.emplace_back(position == numComponents-1 ? 1 : myComponent.dimensions.back()+otherComponent.dimensions.back());
			
			const Tensor::Representation newRep = myComponent.is_sparse() && otherComponent.is_sparse() ? Tensor::Representation::Sparse : Tensor::Representation::Dense;
			std::unique_ptr<Tensor> newComponent(new Tensor(std::move(nxtDimensions), newRep));
			
			newComponent->offset_add(myComponent, isOperator ? std::vector<size_t>({0,0,0,0}) : std::vector<size_t>({0,0,0}));
			
			const size_t leftOffset = position == 0 ? 0 : myComponent.dimensions.front();
			const size_t rightOffset = position == numComponents-1 ? 0 : myComponent.dimensions.back();
			
			newComponent->offset_add(otherComponent, isOperator ? std::vector<size_t>({leftOffset,0,0,rightOffset}) : std::vector<size_t>({leftOffset,0,rightOffset}));
			
			set_component(position, std::move(*newComponent));
		}
		XERUS_PA_END("ADD/SUB", "TTNetwork ADD/SUB", std::string("Dims:")+misc::to_string(dimensions)+" Ranks: "+misc::to_string(ranks()));
		
		if(initialCanonicalization) {
			move_core(initialCorePosition);
		}
		
		return *this;
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator>& TTNetwork<isOperator>::operator-=(const TTNetwork<isOperator>& _other) {
		operator*=(-1.0);
		operator+=(_other);
		operator*=(-1.0);
		return *this;
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::operator*=(const value_t _factor) {
		REQUIRE(!nodes.empty(), "There must not be a TTNetwork without any node");
		
		if(canonicalized) {
			component(corePosition) *= _factor;
		} else {
			component(0) *= _factor;
		}
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::operator/=(const value_t _divisor) {
		operator*=(1/_divisor);
	}
	
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
	
	
	
	template<>
	bool TTNetwork<false>::specialized_contraction_f(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& /*unused*/, internal::IndexedTensorReadOnly<TensorNetwork>&& /*unused*/, internal::IndexedTensorReadOnly<TensorNetwork>&& /*unused*/) {
		// Only TTOperators construct stacks, so no specialized contractions for TTTensors
		return false;
	}
	
	template<>
	bool TTNetwork<true>::specialized_contraction_f(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& _out, internal::IndexedTensorReadOnly<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		_me.assign_indices();
		_other.assign_indices();
		
		const TTNetwork* const meTT = dynamic_cast<const TTNetwork*>(_me.tensorObjectReadOnly);
		const internal::TTStack<true>* const meTTStack = dynamic_cast<const internal::TTStack<true>*>(_me.tensorObjectReadOnly);
		INTERNAL_CHECK(meTT || meTTStack, "Internal Error.");
		
		const TTTensor* const otherTT = dynamic_cast<const TTTensor*>(_other.tensorObjectReadOnly);
		const internal::TTStack<false>* const otherTTStack = dynamic_cast<const internal::TTStack<false>*>(_other.tensorObjectReadOnly);
		const TTOperator* const otherTTO = dynamic_cast<const TTOperator*>(_other.tensorObjectReadOnly);
		const internal::TTStack<true>* const otherTTOStack = dynamic_cast<const internal::TTStack<true>*>(_other.tensorObjectReadOnly);
		
		if ((otherTT == nullptr) && (otherTTStack == nullptr) && (otherTTO == nullptr) && (otherTTOStack == nullptr)) {
			return false;
		}
		
		bool cannoAtTheEnd = false;
		size_t coreAtTheEnd = 0;
		if (meTT != nullptr) {
			cannoAtTheEnd = meTT->canonicalized;
			coreAtTheEnd = meTT->corePosition;
		} else {
			cannoAtTheEnd = meTTStack->cannonicalization_required;
			coreAtTheEnd = meTTStack->futureCorePosition;
		}
		
		
		// TODO profiler should warn if other->corePosition is not identical to coreAtTheEnd
		
		// Determine my first half and second half of indices
		auto midIndexItr = _me.indices.begin();
		size_t spanSum = 0;
		while (spanSum < _me.degree() / 2) {
			INTERNAL_CHECK(midIndexItr != _me.indices.end(), "Internal Error.");
			spanSum += midIndexItr->span;
			++midIndexItr;
		}
		if (spanSum > _me.degree() / 2) {
			return false; // an index spanned some links of the left and some of the right side
		}
		
		if ((otherTT != nullptr) || (otherTTStack != nullptr)) {
			// ensure fitting indices
			if (std::equal(_me.indices.begin(), midIndexItr, _other.indices.begin()) || std::equal(midIndexItr, _me.indices.end(), _other.indices.begin())) {
				_out.reset(new internal::IndexedTensorMoveable<TensorNetwork>(new internal::TTStack<false>(cannoAtTheEnd, coreAtTheEnd), _me.indices));
				*_out->tensorObject = *_me.tensorObjectReadOnly;
				TensorNetwork::add_network_to_network(std::move(*_out), std::move(_other));
				return true;
			} 
				return false;
			
		} else { // other is operator or operator stack
			// determine other middle index
			auto otherMidIndexItr = _other.indices.begin();
			spanSum = 0;
			while (spanSum < _other.degree() / 2) {
				INTERNAL_CHECK(otherMidIndexItr != _other.indices.end(), "Internal Error.");
				spanSum += otherMidIndexItr->span;
				++otherMidIndexItr;
			}
			if (spanSum > _other.degree() / 2) {
				return false; // an index spanned some links of the left and some of the right side
			}
			// or indices in fitting order to contract the TTOs
			if (   std::equal(_me.indices.begin(), midIndexItr, _other.indices.begin()) 
				|| std::equal(midIndexItr, _me.indices.end(), _other.indices.begin())
				|| std::equal(_me.indices.begin(), midIndexItr, otherMidIndexItr) 
				|| std::equal(midIndexItr, _me.indices.end(), otherMidIndexItr)) 
			{
				_out.reset(new internal::IndexedTensorMoveable<TensorNetwork>(new internal::TTStack<true>(cannoAtTheEnd, coreAtTheEnd), _me.indices));
				*_out->tensorObject = *_me.tensorObjectReadOnly;
				TensorNetwork::add_network_to_network(std::move(*_out), std::move(_other));
				return true;
			} 
				return false;
			
		}
	}
	
	
	template<bool isOperator>
	void transpose_if_operator(TTNetwork<isOperator>& _ttNetwork);
	
	template<>
	void transpose_if_operator<false>(TTNetwork<false>&  /*_ttNetwork*/) {}
	
	template<>
	void transpose_if_operator<true>(TTNetwork<true>& _ttNetwork) {
		_ttNetwork.transpose();
	}
	
	template<bool isOperator>
	bool TTNetwork<isOperator>::specialized_sum_f(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& _out, internal::IndexedTensorReadOnly<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		_me.assign_indices();
		_other.assign_indices();
		
		// If the other is not a TT tensor (or stack) fall back to default summation (i.e. return false)
		const TTNetwork* otherTT = dynamic_cast<const TTNetwork*>( _other.tensorObjectReadOnly);
		const internal::TTStack<isOperator>* otherTTStack = dynamic_cast<const internal::TTStack<isOperator>*>( _other.tensorObjectReadOnly);
		if (!otherTT && !otherTTStack) { return false; }
				
		bool transposeRHS;
		if(_me.indices == _other.indices) { // Everything is easy.
			REQUIRE(_me.tensorObjectReadOnly->dimensions == _other.tensorObjectReadOnly->dimensions, "TT sum requires both operants to share the same dimensions.");
			transposeRHS = false;
		} else if (isOperator) { // Check for transposition
			// Find index mid-points to compare the halves separately
			auto myMidIndexItr = _me.indices.begin();
			size_t spanSum = 0;
			while (spanSum < _me.degree() / 2) {
				spanSum += myMidIndexItr->span;
				++myMidIndexItr;
			}
			
			auto otherMidIndexItr = _other.indices.begin();
			spanSum = 0;
			while (spanSum < _other.degree() / 2) {
				spanSum += otherMidIndexItr->span;
				++otherMidIndexItr;
			}
			
			if(std::equal(_me.indices.begin(), myMidIndexItr, otherMidIndexItr) && std::equal(myMidIndexItr, _me.indices.end(), _other.indices.begin())) {
				transposeRHS = true;
			} else {
				return false;
			}
		} else {
			return false; // Not Operator and index order differs.
		}
		
		// Check whether we are a TTStack
		std::unique_ptr<TTNetwork<isOperator>> meStorage;
		const TTNetwork* usedMe;
		
		internal::IndexedTensorMoveable<TensorNetwork>* const moveMe = dynamic_cast<internal::IndexedTensorMoveable<TensorNetwork>*>(&_me);
		internal::TTStack<isOperator>* stackMe;
		if(moveMe && (stackMe = dynamic_cast<internal::TTStack<isOperator>*>(moveMe->tensorObject))) {
			meStorage.reset(new TTNetwork());
			usedMe = meStorage.get();
			*meStorage = TTNetwork(*stackMe);
			INTERNAL_CHECK(usedMe->dimensions == stackMe->dimensions, "Ie " << stackMe->dimensions << " vs "  << usedMe->dimensions);
		} else { // I am normal
			INTERNAL_CHECK(dynamic_cast<const TTNetwork<isOperator>*>(_me.tensorObjectReadOnly),"Non-moveable TTStack (or other error) detected.");
			usedMe = static_cast<const TTNetwork<isOperator>*>(_me.tensorObjectReadOnly);
		}
		const TTNetwork& ttMe = *usedMe;
		
		
		// Check whether the other is a TTStack
		TTNetwork ttOther;
		
		internal::IndexedTensorMoveable<TensorNetwork>* const moveOther = dynamic_cast<internal::IndexedTensorMoveable<TensorNetwork>*>(&_other);
		internal::TTStack<isOperator>* stackOther;
		if(moveOther && (stackOther = dynamic_cast<internal::TTStack<isOperator>*>(moveOther->tensorObject))) {
			ttOther = TTNetwork(*stackOther);
			INTERNAL_CHECK(ttOther.dimensions == stackOther->dimensions, "Ie");
		} else { // Other is normal
			INTERNAL_CHECK(dynamic_cast<const TTNetwork<isOperator>*>(_other.tensorObjectReadOnly),"Non-moveable TTStack (or other error) detected.");
			ttOther = *static_cast<const TTNetwork<isOperator>*>(_other.tensorObjectReadOnly);
			
		}
		
		if(transposeRHS) {
			transpose_if_operator(ttOther);
		}
		
		_out.reset(new internal::IndexedTensorMoveable<TensorNetwork>( new TTNetwork(ttMe), _me.indices));
		
		*static_cast<TTNetwork*>(_out->tensorObject) += ttOther;
		return true;
	}
	
	
	template<bool isOperator>
	void TTNetwork<isOperator>::specialized_evaluation(internal::IndexedTensorWritable<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		INTERNAL_CHECK(_me.tensorObject == this, "Internal Error.");
		
		_me.assign_indices(_other.degree());
		_other.assign_indices();
		const size_t numComponents = _other.degree()/N;
		TTNetwork& meTTN = static_cast<TTNetwork&>(*_me.tensorObject);
		
		// First check whether the other is a TTNetwork as well, otherwise we can skip to fallback
		const TTNetwork* const otherTTN = dynamic_cast<const TTNetwork*>(_other.tensorObjectReadOnly);
		const internal::TTStack<isOperator>* const otherTTStack = dynamic_cast<const internal::TTStack<isOperator>*>(_other.tensorObjectReadOnly);
		internal::IndexedTensorMoveable<TensorNetwork> *movOther = dynamic_cast<internal::IndexedTensorMoveable<TensorNetwork> *>(&_other);
		
		if(otherTTN || otherTTStack) {
			if (otherTTStack) {
				INTERNAL_CHECK(movOther, "Not moveable TTStack encountered...");
				internal::TTStack<isOperator>::contract_stack(std::move(*movOther));
			}
			
			// Check whether the index order coincides
			if (_me.indices == _other.indices) {
				if (otherTTN) {
					meTTN = *otherTTN;
				} else {
					_me.tensorObject->operator=(*_other.tensorObjectReadOnly);
					meTTN.canonicalized = false;
					if (otherTTStack->cannonicalization_required) {
						meTTN.move_core(otherTTStack->futureCorePosition);
					}
				}
				return;
			}
			
			// For TTOperators also check whether the index order is transposed
			if (isOperator) {
				bool transposed = false;
				
				auto midIndexItr = _me.indices.begin();
				size_t spanSum = 0;
				while (spanSum < numComponents) {
					INTERNAL_CHECK(midIndexItr != _me.indices.end(), "Internal Error.");
					spanSum += midIndexItr->span;
					++midIndexItr;
				}
				if (spanSum == numComponents) {
					// Transposition possible on my end
					auto otherMidIndexItr = _other.indices.begin();
					spanSum = 0;
					while (spanSum < numComponents) {
						INTERNAL_CHECK(otherMidIndexItr != _other.indices.end(), "Internal Error.");
						spanSum += otherMidIndexItr->span;
						++otherMidIndexItr;
					}
					if (spanSum == numComponents) {
						// Other tensor also transposable
						transposed = (std::equal(_me.indices.begin(), midIndexItr, otherMidIndexItr)) 
						&& (std::equal(midIndexItr, _me.indices.end(), _other.indices.begin()));
					}
				}
				
				if (transposed) {
					if (otherTTN) {
						meTTN = *otherTTN;
					} else {
						_me.tensorObject->operator=(*_other.tensorObjectReadOnly);
						meTTN.canonicalized = false;
						if (otherTTStack->cannonicalization_required) {
							meTTN.move_core(otherTTStack->futureCorePosition);
						}
					}
					require_correct_format();
					dynamic_cast<TTOperator*>(_me.tensorObject)->transpose(); // NOTE will never be called if !isOperator
					return;
				}
			}
		}
		
		// Use Tensor fallback
		if (_other.tensorObjectReadOnly->nodes.size() > 1) {
			LOG_ONCE(warning, "Assigning a general tensor network to TTOperator not yet implemented. casting to fullTensor first");
		}
		Tensor otherFull(*_other.tensorObjectReadOnly);
		Tensor otherReordered;
		otherReordered(_me.indices) = otherFull(_other.indices);
		
		// Cast to TTNetwork
		*_me.tensorObject = TTNetwork(std::move(otherReordered));
	}
	
	
	// Explicit instantiation of the two template parameters that will be implemented in the xerus library
	template class TTNetwork<false>;
	template class TTNetwork<true>;
	
	
	
	template<bool isOperator>
	TTNetwork<isOperator> operator+(TTNetwork<isOperator> _lhs, const TTNetwork<isOperator>& _rhs) {
		_lhs += _rhs; // NOTE pass-by-value!
		return _lhs;
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator> operator-(TTNetwork<isOperator> _lhs, const TTNetwork<isOperator>& _rhs) {
		_lhs -= _rhs; // NOTE pass-by-value!
		return _lhs;
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator> operator*(TTNetwork<isOperator> _network, const value_t _factor) {
		_network *= _factor; // NOTE pass-by-value!
		return _network;
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator> operator*(const value_t _factor, TTNetwork<isOperator> _network) {
		_network *= _factor; // NOTE pass-by-value!
		return _network;
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator> operator/(TTNetwork<isOperator> _network, const value_t _divisor) {
		_network /= _divisor; // NOTE pass-by-value!
		return _network;
	}
	
	// Explicit instantiation for both types
	template TTNetwork<false> operator+(TTNetwork<false> _lhs, const TTNetwork<false>& _rhs);
	template TTNetwork<true> operator+(TTNetwork<true> _lhs, const TTNetwork<true>& _rhs);
	template TTNetwork<false> operator-(TTNetwork<false> _lhs, const TTNetwork<false>& _rhs);
	template TTNetwork<true> operator-(TTNetwork<true> _lhs, const TTNetwork<true>& _rhs);
	template TTNetwork<false> operator*(TTNetwork<false> _network, const value_t _factor);
	template TTNetwork<true> operator*(TTNetwork<true> _network, const value_t _factor);
	template TTNetwork<false> operator*(const value_t _factor, TTNetwork<false> _network);
	template TTNetwork<true> operator*(const value_t _factor, TTNetwork<true> _network);
	template TTNetwork<false> operator/(TTNetwork<false> _network, const value_t _divisor);
	template TTNetwork<true> operator/(TTNetwork<true> _network, const value_t _divisor);
	
	
	
	
	template<bool isOperator>
	void perform_component_product(Tensor& _newComponent, const Tensor& _componentA, const Tensor& _componentB) {
		const size_t externalDim = isOperator ? _componentA.dimensions[1] * _componentA.dimensions[2] : _componentA.dimensions[1];
		
		if(_componentA.is_dense() && _componentB.is_dense()) {
			INTERNAL_CHECK(_newComponent.is_dense(), "IE");
			value_t* const newCompData = _newComponent.get_dense_data();
			const value_t* const compBData = _componentB.get_unsanitized_dense_data();
			for (size_t r1 = 0; r1 < _componentA.dimensions.front(); ++r1) {
				for (size_t s1 = 0; s1 < _componentB.dimensions.front(); ++s1) {
					for (size_t n = 0; n < externalDim; ++n) {
						for (size_t r2 = 0; r2 < _componentA.dimensions.back(); ++r2) {
							const size_t offsetA = (r1*externalDim + n)*_componentA.dimensions.back()+r2;
							const size_t offsetB = (s1*externalDim + n)*_componentB.dimensions.back();
							const size_t offsetResult = (((r1*_componentB.dimensions.front() + s1)*externalDim + n)*_componentA.dimensions.back()+r2)*_componentB.dimensions.back();
							misc::copy_scaled(newCompData+offsetResult, _componentB.factor*_componentA[offsetA], compBData+offsetB, _componentB.dimensions.back());
						}
					}
				}
			}
		} else if(_componentA.is_dense()) { // B sparse
			const std::vector<std::vector<std::tuple<size_t, size_t, value_t>>> groupedEntriesB = get_grouped_entries<isOperator>(_componentB);
			value_t* const newCompData = _newComponent.get_dense_data();
			for (size_t r1 = 0; r1 < _componentA.dimensions.front(); ++r1) {
				for (size_t n = 0; n < externalDim; ++n) {
					for (size_t r2 = 0; r2 < _componentA.dimensions.back(); ++r2) {
						for(const std::tuple<size_t, size_t, value_t>& entryB : groupedEntriesB[n]) {
							const size_t offsetA = (r1*externalDim + n)*_componentA.dimensions.back()+r2;
							const size_t offsetResult = (((r1*_componentB.dimensions.front() + std::get<0>(entryB))*externalDim + n)*_componentA.dimensions.back()+r2)*_componentB.dimensions.back()+std::get<1>(entryB);
							newCompData[offsetResult] = _componentA[offsetA]*std::get<2>(entryB);
						}
					}
				}
			}
		} else if(_componentB.is_dense()) { // A sparse
			LOG(woot, "");
			value_t* const newCompData = _newComponent.get_dense_data();
			const value_t* const compBData = _componentB.get_unsanitized_dense_data();
			for(const auto& entryA : _componentA.get_unsanitized_sparse_data()) {
				const size_t r2 = entryA.first%_componentA.dimensions.back();
				const size_t n = (entryA.first/_componentA.dimensions.back())%externalDim;
				const size_t r1 = (entryA.first/_componentA.dimensions.back())/externalDim;
				
				for (size_t s1 = 0; s1 < _componentB.dimensions.front(); ++s1) {
					const size_t offsetB = (s1*externalDim + n)*_componentB.dimensions.back();
					const size_t offsetResult = (((r1*_componentB.dimensions.front() + s1)*externalDim + n)*_componentA.dimensions.back()+r2)*_componentB.dimensions.back();
					misc::copy_scaled(newCompData+offsetResult, _componentB.factor*_componentA.factor*entryA.second, compBData+offsetB, _componentB.dimensions.back());
				}
			}
		} else {
			const std::vector<std::vector<std::tuple<size_t, size_t, value_t>>> groupedEntriesB = get_grouped_entries<isOperator>(_componentB);
			std::map<size_t, value_t>& dataMap = _newComponent.get_sparse_data();
			INTERNAL_CHECK(dataMap.empty(), "IE");
			for(const auto& entryA : _componentA.get_unsanitized_sparse_data()) {
				const size_t r2 = entryA.first%_componentA.dimensions.back();
				const size_t n = (entryA.first/_componentA.dimensions.back())%externalDim;
				const size_t r1 = (entryA.first/_componentA.dimensions.back())/externalDim;				
				
				for(const std::tuple<size_t, size_t, value_t>& entryB : groupedEntriesB[n]) {
					dataMap.emplace((((r1*_componentB.dimensions.front() + std::get<0>(entryB))*externalDim + n)*_componentA.dimensions.back()+r2)*_componentB.dimensions.back()+std::get<1>(entryB), _componentA.factor*entryA.second*std::get<2>(entryB));
				}
			}
		}
	}
	
	template<bool isOperator>
	TTNetwork<isOperator> entrywise_product(const TTNetwork<isOperator> &_A, const TTNetwork<isOperator> &_B) {
		static constexpr const size_t N = isOperator?2:1;
		REQUIRE(_A.dimensions == _B.dimensions, "Entrywise_product ill-defined for different external dimensions.");
		
		if(_A.degree() == 0) {
			TTNetwork<isOperator> result(_A);
			result *= _B[0];
			return result;
		}
		
		TTNetwork<isOperator> result(_A.degree());
		const size_t numComponents = _A.degree() / N;
		
		#pragma omp for schedule(static)
		for (size_t i = 0; i < numComponents; ++i) {
			const Tensor& componentA = _A.get_component(i);
			const Tensor& componentB = _B.get_component(i);
			const Tensor::Representation newRep = componentA.is_sparse() && componentB.is_sparse() ? Tensor::Representation::Sparse : Tensor::Representation::Dense;
			Tensor newComponent(isOperator ? 
				Tensor::DimensionTuple({componentA.dimensions.front()*componentB.dimensions.front(), componentA.dimensions[1], componentA.dimensions[2], componentA.dimensions.back()*componentB.dimensions.back()}) : 
				Tensor::DimensionTuple({componentA.dimensions.front()*componentB.dimensions.front(), componentA.dimensions[1], componentA.dimensions.back()*componentB.dimensions.back()}), newRep);
			
			perform_component_product<isOperator>(newComponent, componentA, componentB);
			
			#pragma omp critical
			{
				result.set_component(i, std::move(newComponent));
			}
		}
		
		if (_A.canonicalized && _B.canonicalized) {
			result.move_core(_A.corePosition);
		}
		return result;
	}
	
	
	//Explicit instantiation for both types
	template TTNetwork<false> entrywise_product(const TTNetwork<false> &_A, const TTNetwork<false> &_B);
	template TTNetwork<true> entrywise_product(const TTNetwork<true> &_A, const TTNetwork<true> &_B);
	
	
	
	template<bool isOperator>
	TTNetwork<isOperator> dyadic_product(const TTNetwork<isOperator> &_lhs, const TTNetwork<isOperator> &_rhs) {
		constexpr size_t N = isOperator?2:1;
		_lhs.require_correct_format();
		_rhs.require_correct_format();
		
		if (_lhs.degree() == 0) {
			TTNetwork<isOperator> result(_rhs);
			result *= _lhs[0];
			return result;
		}
		
		TTNetwork<isOperator> result(_lhs);
		if (_rhs.degree() == 0) {
			result *= _rhs[0];
			return result;
		}
		
		const size_t lhsNumComponents = _lhs.degree()/N;
		const size_t rhsNumComponents = _rhs.degree()/N;
		
		// fix external links of lhs nodes
		for (size_t i=1; i<result.nodes.size(); ++i) {
			for (TensorNetwork::Link &l : result.nodes[i].neighbors) {
				if (l.external) {
					if (l.indexPosition >= lhsNumComponents) {
						l.indexPosition += rhsNumComponents;
					}
				}
			}
		}
		
		// Add all nodes of rhs and fix neighbor relations
		result.nodes.pop_back();
		result.nodes.reserve(_lhs.degree()+_rhs.degree()+2);
		for (size_t i = 1; i < _rhs.nodes.size(); ++i) {
			result.nodes.emplace_back(_rhs.nodes[i]);
			for (TensorNetwork::Link &l : result.nodes.back().neighbors) {
				if (l.external) {
					if (l.indexPosition < rhsNumComponents) {
						l.indexPosition += lhsNumComponents;
					} else {
						l.indexPosition += 2*lhsNumComponents;
					}
				} else {
					if (l.other==0) {
						l.indexPosition = N+1;
					}
					l.other += lhsNumComponents;
				}
			}
		}
		
		// Add all external indices of rhs
		result.externalLinks.clear(); // NOTE that this is necessary because in the operator case we added indices 
		result.dimensions.clear();   //        in the wrong position when we copied the lhs
		result.externalLinks.reserve(_lhs.degree()+_rhs.degree());
		result.dimensions.reserve(_lhs.degree()+_rhs.degree());
		
		for (size_t i = 0; i < lhsNumComponents; ++i) {
			const size_t d=_lhs.dimensions[i];
			result.externalLinks.emplace_back(i+1, 1, d, false);
			result.dimensions.push_back(d);
		}
		
		for (size_t i = 0; i < rhsNumComponents; ++i) {
			const size_t d = _rhs.dimensions[i];
			result.externalLinks.emplace_back(lhsNumComponents+i+1, 1, d, false);
			result.dimensions.push_back(d);
		}
		
		if (isOperator) {
			for (size_t i = 0; i < lhsNumComponents; ++i) {
				const size_t d = _lhs.dimensions[i];
				result.externalLinks.emplace_back(i+1, 2, d, false);
				result.dimensions.push_back(d);
			}
			for (size_t i = 0; i < rhsNumComponents; ++i) {
				const size_t d = _rhs.dimensions[i];
				result.externalLinks.emplace_back(lhsNumComponents+i+1, 2, d, false);
				result.dimensions.push_back(d);
			}
		}
		
		if (_lhs.canonicalized && _rhs.canonicalized) {
			if (_lhs.corePosition == 0 && _rhs.corePosition == 0) {
				result.canonicalized = true;
				result.corePosition = lhsNumComponents;
				// the other core might have carried a factor
				if (result.nodes[1].tensorObject->has_factor()) {
					(*result.nodes[lhsNumComponents+1].tensorObject) *= result.nodes[1].tensorObject->factor;
					result.nodes[1].tensorObject->factor = 1.0;
				}
				result.move_core(0);
			} else if (_lhs.corePosition == lhsNumComponents-1 && _rhs.corePosition == rhsNumComponents-1) {
				result.canonicalized = true;
				result.corePosition = lhsNumComponents-1;
				const size_t lastIdx = lhsNumComponents + rhsNumComponents -1;
				// the other core might have carried a factor
				if (result.nodes[lastIdx+1].tensorObject->has_factor()) {
					(*result.nodes[lhsNumComponents].tensorObject) *= result.nodes[lastIdx+1].tensorObject->factor;
					result.nodes[lastIdx+1].tensorObject->factor = 1.0;
				}
				result.move_core(lastIdx);
			}
		} else {
			result.canonicalized = false;
		}
		
		result.require_correct_format();
		return result;
	}
	
	template TTNetwork<true> dyadic_product(const TTNetwork<true> &_lhs, const TTNetwork<true> &_rhs);
	template TTNetwork<false> dyadic_product(const TTNetwork<false> &_lhs, const TTNetwork<false> &_rhs);
	
	template<bool isOperator>
	TTNetwork<isOperator> dyadic_product(const std::vector<TTNetwork<isOperator>>& _tensors) {
		if (_tensors.empty()) { return TTNetwork<isOperator>(); }
		
		TTNetwork<isOperator> result(_tensors.back());
		// construct dyadic products right to left as default cannonicalization is left
		for (size_t i = _tensors.size()-1; i > 0; --i) {
			XERUS_REQUIRE_TEST;
			result = dyadic_product(_tensors[i-1], result);
		}
		return result;
	}
	
	template TTNetwork<true> dyadic_product(const std::vector<TTNetwork<true>>& _tensors);
	template TTNetwork<false> dyadic_product(const std::vector<TTNetwork<false>>& _tensors);
	
	
	
	namespace misc {
		
		template<bool isOperator>
		void stream_writer(std::ostream& _stream, const TTNetwork<isOperator> &_obj, misc::FileFormat _format) {
			if(_format == misc::FileFormat::TSV) {
				_stream << std::setprecision(std::numeric_limits<value_t>::digits10 + 1);
			}
			// storage version number
			write_to_stream<size_t>(_stream, 1, _format);
			
			// store TN specific data
			write_to_stream<bool>(_stream, _obj.canonicalized, _format);
            write_to_stream<size_t>(_stream, _obj.corePosition, _format);

			
			// save rest of TN
			write_to_stream<TensorNetwork>(_stream, _obj, _format);
		}
		template void stream_writer(std::ostream& _stream, const TTNetwork<true> &_obj, misc::FileFormat _format);
		template void stream_writer(std::ostream& _stream, const TTNetwork<false> &_obj, misc::FileFormat _format);
		
		
		template<bool isOperator>
		void stream_reader(std::istream& _stream, TTNetwork<isOperator> &_obj, const misc::FileFormat _format) {
			IF_CHECK( size_t ver = ) read_from_stream<size_t>(_stream, _format);
			REQUIRE(ver == 1, "Unknown stream version to open (" << ver << ")");
			
			// load TN specific data
			read_from_stream<bool>(_stream, _obj.canonicalized, _format);
            read_from_stream<size_t>(_stream, _obj.corePosition, _format);

			
			// load rest of TN
			read_from_stream<TensorNetwork>(_stream, _obj, _format);
			
			_obj.require_correct_format();
		}
		template void stream_reader(std::istream& _stream, TTNetwork<true> &_obj, const misc::FileFormat _format);
		template void stream_reader(std::istream& _stream, TTNetwork<false> &_obj, const misc::FileFormat _format);
	} // namespace misc
	
} // namespace xerus

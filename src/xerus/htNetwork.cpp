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
* @brief Implementation of the HTNetwork class (and thus HTTensor and HTOperator).
*/

#include <algorithm>
#include <memory>

#include <xerus/htNetwork.h>

#include <xerus/misc/check.h>
#include <xerus/misc/math.h>
#include <xerus/misc/performanceAnalysis.h>
#include <xerus/misc/internal.h>

#include <xerus/basic.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/htStack.h>
#include <xerus/indexedTensorList.h>
#include <xerus/indexedTensorMoveable.h>

namespace xerus {
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	template<bool isOperator>
	HTNetwork<isOperator>::HTNetwork() : TensorNetwork(), canonicalized(false) {}
	
	
	template<bool isOperator>
	HTNetwork<isOperator>::HTNetwork(const Tensor& _tensor, const double _eps, const size_t _maxRank) :
		HTNetwork(_tensor, _eps, std::vector<size_t>(_tensor.degree() <= 1 ? 0 : 2 * _tensor.degree()/N - 2,_maxRank)) {}

	
	template<bool isOperator>
	HTNetwork<isOperator>::HTNetwork(const size_t _degree) : HTNetwork(std::vector<size_t>(_degree, 1)) { }
	
	template<bool isOperator>
	HTNetwork<isOperator>::HTNetwork(Tensor::DimensionTuple _dimensions) : TensorNetwork(ZeroNode::None), canonicalized(true), corePosition(0) {
		dimensions = std::move(_dimensions);
		REQUIRE(dimensions.size()%N==0, "Illegal degree for HTOperator.");
		REQUIRE(!misc::contains(dimensions, size_t(0)), "Zero is no valid dimension.");
		//Number of Leafs
		const size_t numLeaves = dimensions.size()/N;
		//Lvl, NOTE: adding 0.5 to overcome possible conversion errors
		const size_t numFullLeaves = numLeaves == 1 ? 2 : static_cast<size_t>(0.5+std::pow(2,std::ceil(std::log2(static_cast<double>(numLeaves)))));
		// Number of internal components
		const size_t numIntCom = numLeaves - 1;
		// set number of components
		// = numLeaves + numIntCom; //TODO rethink this

		if (numLeaves == 0) {
			nodes.emplace_back(std::make_unique<Tensor>());
			return;
		}
		std::vector<size_t> leaf_order;
		for (size_t i = numFullLeaves - 1; i < numIntCom + numLeaves; ++i)
			leaf_order.emplace_back(i-numIntCom);
		for (size_t i = numLeaves - 1; i < numFullLeaves - 1 ; ++i)
			leaf_order.emplace_back(i-numIntCom);
		// ExternalLinks
		externalLinks.reserve(numLeaves * N);
		size_t ii = 0;
		for (size_t i : leaf_order) {
			externalLinks.emplace_back(i + numIntCom, 1, dimensions[ii], false);
			ii++;
		}
		if (isOperator) {
			ii = 0;
			for (size_t i : leaf_order) {
				externalLinks.emplace_back(i + numIntCom, 2, dimensions[numLeaves+ ii], false);
				ii++;
			}
		}
		//Add dummy at root of tree
		//externalLinks.emplace_back(2*numFullLeaves, 0, 1, false);
		
		std::vector<TensorNetwork::Link> neighbors;
		
		//neighbors.emplace_back(1, 0, 1, false);
		
		//Loop for the internal nodes
		for ( size_t i = 0; i < numIntCom; ++i){
			neighbors.clear();
			//parent
			neighbors.emplace_back(i==0 ? 2*numLeaves - 1: (i+1) / 2 - 1, i==0 ? 0 : (i+1)%2 + 1, 1, false);
			//child 1
			neighbors.emplace_back(2*i + 1, 0, 1, false);
			//child 2
			neighbors.emplace_back(2*i + 2, 0, 1, false);
			//node
			nodes.emplace_back( std::make_unique<Tensor>(Tensor::dirac({1, 1, 1}, 0)), std::move(neighbors) );
		}

		//nodes.emplace_back(std::make_unique<Tensor>(Tensor::ones({1})), std::move(neighbors));
		//Loop for the leafs
		for (size_t i = 0; i < numLeaves; ++i) {
			neighbors.clear();
			std::vector<size_t>::iterator itr = std::find(leaf_order.begin(), leaf_order.end(), i);
			size_t leaf_order_pos = static_cast<size_t>(std::distance(leaf_order.begin(), itr));
			//The leafs are the last nodes, numIntCom .. numIntCom + numFullLeaves
			//A parent node is calculated by ((numIntCom + i)+1) / 2 -1 , where i is the linear position in the node vector
			//The parent nodes first index is its parent node 0 then first child 1 then the second child 2
			//This means from a leafs perspective it is mod 2, which translate wrt i to i%2 + 1
			neighbors.emplace_back(((numIntCom + i) + 1) / 2 - 1, (i+numIntCom-1)%2 + 1, 1, false);

			// First come the external dimension then comes the internal dimension for the leaf
			// -1 one is a placeholder for external nodes
			neighbors.emplace_back(-1, leaf_order_pos, dimensions[leaf_order_pos], true);
			if(isOperator) { neighbors.emplace_back(-1, numLeaves + leaf_order_pos, dimensions[numLeaves + leaf_order_pos], true); }

			if(!isOperator) {
				nodes.emplace_back( std::make_unique<Tensor>(Tensor::dirac({1, dimensions[leaf_order_pos]}, 0)), std::move(neighbors) );
			} else {
				nodes.emplace_back( std::make_unique<Tensor>(Tensor::dirac({1, dimensions[leaf_order_pos], dimensions[numLeaves + leaf_order_pos]}, 0)), std::move(neighbors) );
			}
		}
		//Dummy Leaves
//		for (size_t i = numLeaves; i < numFullLeaves; ++i){
//			neighbors.clear();
//			neighbors.emplace_back(((numIntCom + i) + 1) / 2 - 1, i%2 + 1, 1, false);
//			nodes.emplace_back( std::make_unique<Tensor>(Tensor::dirac({1}, 0)), std::move(neighbors) );
//		}

		neighbors.clear();
		neighbors.emplace_back(0, 0, 1, false);
		nodes.emplace_back( std::make_unique<Tensor>(Tensor::ones({1})), std::move(neighbors));
		
		// Make a Zero Tensor (at core)
		//(*nodes[1].tensorObject)[0] = 0;
	}
	

	template<bool isOperator>
	HTNetwork<isOperator>::HTNetwork(const Tensor& _tensor, const double _eps, const TensorNetwork::RankTuple& _maxRanks): HTNetwork(_tensor.degree()) {
		REQUIRE(_tensor.degree()%N==0, "Number of indices must be even for HTOperator");
		REQUIRE(_eps >= 0 && _eps < 1, "_eps must be positive and smaller than one. " << _eps << " was given.");
		REQUIRE(_maxRanks.size() == num_ranks(), "We need " << num_ranks() <<" ranks but " << _maxRanks.size() << " where given");
		REQUIRE(!misc::contains(_maxRanks, size_t(0)), "Maximal ranks must be strictly positive. Here: " << _maxRanks);


		const size_t numExternalComponent = degree()/N;
		size_t numComp = get_number_of_components();
		size_t lvl = static_cast<size_t>(0.5 + std::floor(std::log2(static_cast<double>(numComp - numExternalComponent))));
		size_t numCompOnLvl = static_cast<size_t>(0.5+std::pow(2.,static_cast<double>(lvl)));
		if (_tensor.degree() == 0) {
			*nodes[0].tensorObject = _tensor;
			return;
		}
		dimensions = _tensor.dimensions;
		Tensor remains;



		if(isOperator) {
			//For operators the index pairs (i,i+1) for i%2=0 become the ith in the first and second half of the indexes
			//i.e. for 0,1,2,3,4,5,6,7 goes to 0,2,4,6,1,3,5,7
			std::vector<size_t> shuffle(_tensor.degree());
			for(size_t i = 0; i < numExternalComponent; ++i) {
				shuffle[i] = 2*i;
				shuffle[numExternalComponent + i] = 2*i+1;
			}
			xerus::reshuffle(remains, _tensor, shuffle);
		} else {
			remains = _tensor;
		}

		Tensor singularValues, newNode;
		//for the leaves
		std::vector<size_t> leaf_order;
		for (size_t i = 2*numCompOnLvl - 1; i < numComp; ++i)
			leaf_order.emplace_back(i);
		for (size_t i = numExternalComponent - 1; i < 2*numCompOnLvl - 1 ; ++i)
			leaf_order.emplace_back(i);

		size_t ii = 0;
		for(size_t pos: leaf_order) {
			std::vector<size_t> ithmode(remains.degree());
			std::vector<size_t> ithmodeinv(remains.degree() - 1);

			if(isOperator) {
				size_t lengthrem = remains.degree();
				for(size_t j = 0; j < lengthrem ; ++j) {
					ithmode[j] = j == ii ? 0 : (j== ii + 1 ? 1 : j < ii ? j + 2 : j );
				}
				for(size_t j = 0; j < lengthrem - 1; ++j) {
					ithmodeinv[j] = j == 0 ? ii : (j <= ii ? j - 1 : j);
				}
			} else {
				for(size_t j = 0; j < numExternalComponent; ++j) {
					ithmode[j] = j == 0 ? ii : (j == ii ? 0 : j);
				}
			}

			xerus::reshuffle(remains, remains, ithmode);

			calculate_svd(newNode, singularValues, remains, remains, N, _maxRanks[pos - 1], _eps);
			if (isOperator){
				xerus::reshuffle(newNode, newNode, {1,2,0});
			} else {
				xerus::reshuffle(newNode, newNode, {1,0});
			}
			set_component(pos, std::move(newNode));
			newNode.reset();
			xerus::contract(remains, singularValues, false, remains, false, 1);

			xerus::reshuffle(remains, remains, isOperator ? ithmodeinv : ithmode);
			ii++;
		}
		//for the internal components
		size_t startposition = 2*numCompOnLvl - numExternalComponent;
		//add dummy dimensions to remainder
//		Tensor::DimensionTuple wdummydim(numCompOnLvl * 2);
//		for (size_t i = 0; i < numCompOnLvl * 2; i++){
//			wdummydim[i] = i < numExternalComponent ? remains.dimensions[i] : 1;
//		}
//		remains.reinterpret_dimensions(wdummydim);
		for(; lvl  > 0; --lvl) {
			for (size_t pos = 0; pos < numCompOnLvl - startposition; ++pos){
				std::vector<size_t> ithmode(remains.degree());
				std::vector<size_t> ithmodeinv(remains.degree() - 1);
				size_t lengthrem = remains.degree();
				for(size_t j = 0; j < lengthrem ; ++j) {
					ithmode[j] = j == pos ? 0 : (j== pos + 1 ? 1 : j < pos ? j + 2 : j );
				}
				for(size_t j = 0; j < lengthrem - 1; ++j) {
					ithmodeinv[j] = j == 0 ? pos : (j <= pos ? j - 1 : j);
				}
				xerus::reshuffle(remains, remains, ithmode);

				calculate_svd(newNode, singularValues, remains, remains, 2, _maxRanks[numCompOnLvl + pos - 2], _eps); // TODO fix maxRanks
				xerus::reshuffle(newNode, newNode, {1,2,0}); // first parent then children
				set_component(numCompOnLvl + pos - 1, std::move(newNode));
				newNode.reset();
				xerus::contract(remains, singularValues, false, remains, false, 1);
				xerus::reshuffle(remains, remains, ithmodeinv);

			}
			numCompOnLvl /= 2;
			startposition = 0;
		}
		Tensor::DimensionTuple wdummydimroot({remains.dimensions[0], remains.dimensions[1], 1});
		remains.reinterpret_dimensions(wdummydimroot);
		xerus::reshuffle(remains, remains, {1,2,0}); // first parent then children
		set_component(0, remains);
		assume_core_position(0);
	}
//
//
//// 	template<bool isOperator>
//// 	TTNetwork<isOperator>::TTNetwork(const TensorNetwork &_network, double _eps) : TTNetwork(Tensor(_network)) {
//// 		LOG(warning, "Cast of arbitrary tensor network to TT not yet supported. Casting to Tensor first"); // TODO
//// 	}
//
//
	template<bool isOperator>
	HTNetwork<isOperator> HTNetwork<isOperator>::ones(const std::vector<size_t>& _dimensions) {
		REQUIRE(_dimensions.size()%N == 0, "Illegal number of dimensions for htOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a HTTensor with dimension 0 is not possible.");

		const size_t numOfLeaves = _dimensions.size()/N;
		const size_t numIntComp = numOfLeaves - 1;
		const size_t numComponents = numIntComp + numOfLeaves;

		if(_dimensions.empty()) {
			return HTNetwork(Tensor::ones({}));
		}

		HTNetwork result(_dimensions.size());
		size_t dim_counter = 2*numOfLeaves - static_cast<size_t>(std::pow(2,std::ceil(std::log2(static_cast<double>(numOfLeaves)))));
		//Leaves
		std::vector<size_t> dimensions(isOperator ? 3 : 2, 1);
		for(size_t i = numIntComp; i < numComponents; ++i,++dim_counter) {
			if (dim_counter == numOfLeaves)
							dim_counter = 0;
			dimensions[1] = _dimensions[dim_counter];
			if (isOperator) {
				dimensions[2] = _dimensions[dim_counter + numOfLeaves];
			}
			result.set_component(i, Tensor::ones(dimensions));
		}
		//Internal Components
		std::vector<size_t> dimensions2(3, 1);
		for(size_t i = 0; i < numIntComp; ++i) {
			result.set_component(i, Tensor::ones(dimensions2));
		}

		result.canonicalize_root();
		return result;
	}


	template<> template<>
	HTNetwork<true> HTNetwork<true>::identity(const std::vector<size_t>& _dimensions) {
		REQUIRE(_dimensions.size()%N==0, "Illegal number of dimensions for htOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a HTTensor with dimension 0 is not possible.");

		if(_dimensions.empty()) {
			return HTNetwork(Tensor::ones({}));
		}

		const size_t numLeaves = _dimensions.size()/N;

		HTNetwork result(_dimensions.size());
		size_t dim_counter = 2*numLeaves - static_cast<size_t>(std::pow(2,std::ceil(std::log2(static_cast<double>(numLeaves)))));
		std::vector<size_t> constructionVector(3, 1);
		for (size_t i = numLeaves-1; i < 2*numLeaves - 1; ++i, ++dim_counter) {
			if (dim_counter == numLeaves)
				dim_counter = 0;
			constructionVector[1] = _dimensions[dim_counter];
			constructionVector[2] = _dimensions[dim_counter + numLeaves];
			result.set_component(i, Tensor(constructionVector, [](const std::vector<size_t> &_idx){
				if (_idx[1] == _idx[2]) {
					return 1.0;
				}
				return 0.0;
			}));
		}

		result.canonicalize_root();
		return result;
	}
//
//	template<bool isOperator>
//	TTNetwork<isOperator> TTNetwork<isOperator>::kronecker(const std::vector<size_t>& _dimensions) {
//		REQUIRE(_dimensions.size()%N == 0, "Illegal number of dimensions for ttOperator");
//		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a TTNetwork with dimension 0 is not possible.");
//
//		if(_dimensions.empty()) {
//			return TTNetwork(Tensor::kronecker({}));
//		}
//
//		TTNetwork result(_dimensions.size());
//		const size_t numNodes = _dimensions.size()/N;
//
//		const auto minN = misc::min(_dimensions);
//
//		// All nodes are simply kronecker tensors themself
//		std::vector<size_t> dimensions;
//		for(size_t i = 0; i < numNodes; ++i) {
//			dimensions.reserve(4);
//			if(i > 0) { dimensions.push_back(minN); }
//			dimensions.push_back(_dimensions[i]);
//			if (isOperator) { dimensions.push_back(_dimensions[i+numNodes]); }
//			if(i+1 < numNodes) { dimensions.push_back(minN); }
//			auto newComp = Tensor::kronecker(dimensions);
//			if(i == 0) { dimensions.insert(dimensions.begin(), 1); }
//			if(i+1 == numNodes) { dimensions.insert(dimensions.end(), 1); }
//			if(i == 0 || i+1 == numNodes) { newComp.reinterpret_dimensions(std::move(dimensions)); }
//			result.set_component(i, std::move(newComp));
//			dimensions.clear();
//		}
//		result.canonicalize_left();
//		return result;
//	}
//
	template<bool isOperator>
	HTNetwork<isOperator> HTNetwork<isOperator>::dirac(std::vector<size_t> _dimensions, const std::vector<size_t>& _position) {
		REQUIRE(_dimensions.size()%N==0, "Illegal number of dimensions for htOperator");
		REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a HTTensor with dimension 0 is not possible.");
		REQUIRE(_dimensions.size() == _position.size(), "Inconsistend number of entries in _dimensions and _position.");

		const size_t numLeaves = _dimensions.size()/N;

		if(numLeaves <= 1) {
			return HTNetwork(Tensor::dirac(_dimensions, _position));
		}

		HTNetwork result(_dimensions);
		size_t dim_counter = 2*numLeaves - static_cast<size_t>(std::pow(2,std::ceil(std::log2(static_cast<double>(numLeaves)))));

		for (size_t i = numLeaves - 1; i < 2*numLeaves-1; ++i, ++dim_counter) {
			if (dim_counter == numLeaves)
				dim_counter = 0;
			if(isOperator) {
				result.set_component(i, Tensor::dirac({1, result.dimensions[dim_counter], result.dimensions[numLeaves+dim_counter]}, _position[dim_counter]*result.dimensions[numLeaves+dim_counter] + _position[numLeaves+dim_counter]));
			} else {
				result.set_component(i, Tensor::dirac({1, result.dimensions[dim_counter]}, _position[dim_counter]));
			}
		}
		return result;
	}

	template<bool isOperator>
	HTNetwork<isOperator> HTNetwork<isOperator>::dirac(std::vector<size_t> _dimensions, const size_t _position) {
		return dirac(_dimensions, Tensor::position_to_multiIndex(_position, _dimensions));
	}

	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

	#ifndef XERUS_DISABLE_RUNTIME_CHECKS
		template<bool isOperator>
		void HTNetwork<isOperator>::require_correct_format() const {
			require_valid_network(); // Network must at least be valid.
			const size_t numLeaves = degree()/N;
			const size_t numNodes = 2*numLeaves - 1;
			REQUIRE(nodes.size() == numNodes + 1, "Wrong number of nodes: " << nodes.size() << " expected " << numNodes << ".");
			REQUIRE(!canonicalized || (degree() == 0 && corePosition == 0) || corePosition < numNodes, "Invalid corePosition: " << corePosition << " there are only " << numNodes << " components.");

			const size_t numFullLeaves = numLeaves == 1 ? 2 : static_cast<size_t>(0.5+std::pow(2,std::ceil(std::log2(static_cast<double>(numLeaves)))));

			std::vector<size_t> leaf_order;
			for (size_t i = numFullLeaves - 1; i < numNodes; ++i)
				leaf_order.emplace_back(i);
			for (size_t i = numLeaves - 1; i < numFullLeaves - 1 ; ++i)
				leaf_order.emplace_back(i);

			// Per external link
			for (size_t n = 0; n < externalLinks.size(); ++n) {
				const TensorNetwork::Link &l = externalLinks[n];
				REQUIRE(l.other == leaf_order[(n%numLeaves)], "The " << n << "-th external link must point the the " << leaf_order[(n%numLeaves)] << "-th component,  but does point to the " << l.other << "-th node.");
			}

			// Virtual nodes
			if(degree() > 0) {
				REQUIRE(nodes.back().degree() == 1, "The virtual node must have degree 1, but has size " << nodes.back().degree());
				REQUIRE(nodes.back().neighbors[0].dimension == 1, "The virtual node's single dimension must be 1, but is " << nodes.back().neighbors[0].dimension);
				REQUIRE(nodes.back().neighbors[0].other == 0, "The virtual node's single link must be to node " << 0 << ", but is towards node " << nodes.back().neighbors[0].other);
				REQUIRE(nodes.back().neighbors[0].indexPosition == 0, "The virtual node's single link must link at indexPosition " << 0 << ", but link at " << nodes.back().neighbors[0].indexPosition);
				REQUIRE(misc::hard_equal((*nodes.back().tensorObject)[0], 1.0), "The virtual node's single entry must be 1.0, but it is " << (*nodes.back().tensorObject)[0]);
				REQUIRE(!nodes.back().tensorObject->has_factor(), "The virtual node must not carry a non-trivial factor.");
			}

			// Per  internal component
			for (size_t n = 0; n < numLeaves-1; ++n) {
				const TensorNode& node = nodes[n];

				REQUIRE(!canonicalized || n == corePosition || !node.tensorObject->has_factor(), "In canonicalized HTNetworks only the core may carry a non-trivial factor. Violated by component " << n << " factor: " << node.tensorObject->factor << " corepos: " << corePosition);

				REQUIRE(node.degree() == 3, "Every internal HT-Component must have degree " << 3 << ", but component " << n << " has degree " << node.degree());
				REQUIRE(!node.neighbors[0].external, "The first link of each internal HT-Component must not be external. Violated by component " << n);
				REQUIRE(node.neighbors[0].other == (n==0? 2*numLeaves - 1 : (n-1)/2), "The first link of each internal HT-Component must link to the parent node. Violated by component " << n << ", which instead links to node " << node.neighbors[0].other << " (expected " << (n==0? 2*numLeaves - 1 : (n-1)/2) << ").");
				REQUIRE(node.neighbors[0].indexPosition == (n==0?0:(is_left_child(n) ? 1 : 2)), "The first link of each HT-Component must link to the correct child index of the parent node. Violated by component " << n << ", which instead links to index " << node.neighbors[0].indexPosition << " (expected " << (n==0?0:(is_left_child(n) ? 1 : 2)) << ").");

				REQUIRE(!node.neighbors[1].external, "The second link of each TT-Component must not be external. Violated by component " << n << ".");
				REQUIRE(node.neighbors[1].other == 2*n + 1, "The second link of each internal HT-Component must link to the left child node. Violated by component " << n << ", which instead links to node " << node.neighbors[0].other << " (expected " << 2*n + 1 << ").");
				REQUIRE(node.neighbors[1].indexPosition == 0, "The second link of each HT-Component must link to the first index of the child node. Violated by component " << n << ", which instead links to index " << node.neighbors[0].indexPosition << " (expected " << 0 << ").");

				REQUIRE(!node.neighbors[2].external, "The third link of each TT-Component must not be external. Violated by component " << n << ".");
				REQUIRE(node.neighbors[2].other == 2*n + 2, "The third link of each internal HT-Component must link to the left child node. Violated by component " << n << ", which instead links to node " << node.neighbors[0].other << " (expected " << 2*n + 2 << ").");
				REQUIRE(node.neighbors[2].indexPosition == 0, "The third link of each HT-Component must link to the first index of the child node. Violated by component " << n << ", which instead links to index " << node.neighbors[0].indexPosition << " (expected " << 0 << ").");

			}
			// Per leaves
			size_t i = 0;
			for (size_t n: leaf_order){
				const TensorNode& node = nodes[n];
				REQUIRE(!canonicalized || n == corePosition || !node.tensorObject->has_factor(), "In canonicalized HTNetworks only the core may carry a non-trivial factor. Violated by component " << n << " factor: " << node.tensorObject->factor << " corepos: " << corePosition);
				REQUIRE(node.degree() == N+1, "Every leaf HT-Component must have degree " << N+1 << ", but component " << n << " has degree " << node.degree());

				REQUIRE(!node.neighbors[0].external, "The first link of each leaf  HT-Component must not be external. Violated by component " << n);
				REQUIRE(node.neighbors[0].other == (n-1)/2, "The first link of each internal HT-Component must link to the parent node. Violated by component " << n << ", which instead links to node " << node.neighbors[0].other << " (expected " << (n-1)/2 << ").");
				REQUIRE(node.neighbors[0].indexPosition == (is_left_child(n) ? 1 : 2), "The first link of each HT-Component must link to the correct child index of the parent node. Violated by component " << n << ", which instead links to index " << node.neighbors[0].indexPosition << " (expected " << (is_left_child(n) ? 1 : 2) << ").");

				REQUIRE(node.neighbors[1].external, "The second link of each HTO-Component must be external. Violated by component " << n << ".");
				REQUIRE(node.neighbors[1].indexPosition == i, "The second link of each HTO-Component must link to the external dimension equal to the leave position. Violated by component " << n << " which links at " << node.neighbors[2].indexPosition << " (expected " << i << ").");

				REQUIRE(!isOperator || node.neighbors[2].external, "The third link of each HTO-Component must be external. Violated by component " << n << ".");
				REQUIRE(!isOperator || node.neighbors[2].indexPosition == i + numLeaves, "The third link of each HTO-Component must link to the external dimension equal to the leave position + numLeaves. Violated by component " << n << " which links at " << node.neighbors[2].indexPosition << " (expected " << i + numLeaves << ").");

				i++;
			}
		}
	#else
		template<bool isOperator>
		void TTNetwork<isOperator>::require_correct_format() const { }
	#endif



	template<bool isOperator>
	size_t HTNetwork<isOperator>::num_ranks() const {
		return degree() == 0 ? 0 : get_number_of_components() - 1;
	}

	template<bool isOperator>
	std::vector<size_t> HTNetwork<isOperator>::get_path(size_t _start, size_t _end) const {
		std::vector<size_t> path_start;
		std::vector<size_t> path_end;
		std::vector<size_t> result;

		REQUIRE(get_path_from_root(0, _start, path_start ), "start point is wrong");
		REQUIRE(get_path_from_root(0, _end, path_end ), "end point is wrong");
		while(!path_start.empty()){
			size_t tmp = path_start.back();
			path_start.pop_back();
			auto tmp_found = std::find(path_end.begin(), path_end.end(), tmp);
			if (path_end.end() == tmp_found){ result.emplace_back(tmp);}
			else{
				result.insert(result.end(), tmp_found, path_end.end());
				break;
			}
		}
		std::reverse(result.begin(),result.end());
		return result;
	}

	template<bool isOperator>
	bool HTNetwork<isOperator>::get_path_from_root(size_t _root, size_t _dest, std::vector<size_t>& _path ) const {
		if (_root > get_number_of_components()) { return false;}
		_path.emplace_back(_root);
		if (_root == _dest) { return true;}
		if(get_path_from_root(_root*2+1,_dest,_path) || get_path_from_root(_root*2+2,_dest,_path)) {return true;}
		_path.pop_back();
		return false;

	}

	template<bool isOperator>
	size_t HTNetwork<isOperator>::get_parent_component(size_t _comp) const{
		REQUIRE(_comp != 0, "The root component has no parent!");
		REQUIRE(_comp <  get_number_of_components() && _comp > 0, "The component requested is out of bounce, given " << _comp);
		return (_comp - 1) / 2;
	}

	template<bool isOperator>
	size_t HTNetwork<isOperator>::get_left_child_component(size_t _comp) const{
		REQUIRE(_comp < get_number_of_components() - degree()/N, "This is a leaf! Leaves do not have children.");
		REQUIRE(_comp <  get_number_of_components() && _comp >= 0, "The component requested is out of bounce, given " << _comp);
		return 2 * _comp + 1;
	}

	template<bool isOperator>
		size_t HTNetwork<isOperator>::get_right_child_component(size_t _comp) const{
			REQUIRE(_comp < get_number_of_components() - degree()/N, "This is a leaf! Leaves do not have children.");
			REQUIRE(_comp <  get_number_of_components() && _comp >= 0, "The component requested is out of bounce, given " << _comp);
			return 2 * _comp + 2;
	}
	template<bool isOperator>
	bool HTNetwork<isOperator>::is_left_child(size_t _comp) const{
		REQUIRE(_comp != 0, "The root component is not a child of another component!");
		REQUIRE(_comp <  get_number_of_components() && _comp > 0, "The component requested is out of bounce, given " << _comp);
		return _comp % 2 == 1;
	}

	template<bool isOperator>
	size_t HTNetwork<isOperator>::get_number_of_components() const{
		const size_t numLeaves = dimensions.size()/N;
		//const size_t numFullLeaves = numLeaves == 1 ? 2 : static_cast<size_t>(0.5+std::pow(2,std::ceil(std::log2(static_cast<double>(numLeaves)))));
		const size_t numIntCom = numLeaves - 1;
		return numLeaves == 0 ? 1 : numLeaves + numIntCom;
	}

	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

//	template<bool isOperator>
//	std::vector<size_t> HTNetwork<isOperator>::reduce_to_maximal_ranks(std::vector<size_t> _ranks, const std::vector<size_t>& _dimensions) {
//		const size_t numComponents = static_cast<size_t>(std::pow(2,std::ceil(std::log2(static_cast<double>(_dimensions.size()/N ))))) - 1 + _dimensions.size()/N;
//		REQUIRE(_dimensions.size()%N == 0, "invalid number of dimensions for HTOperator");
//		REQUIRE(numComponents == _ranks.size()+1, "Invalid number of ranks ("<<_ranks.size()<<") or dimensions ("<<_dimensions.size()<<") given.");
//
//		// Left to right sweep
//		size_t currMax = 1;
//		for (size_t i = 0; i+1 < numComponents; ++i) {
//			currMax *= _dimensions[i];
//			if (isOperator) { currMax *= _dimensions[numComponents+i]; }
//
//			if (currMax < _ranks[i]) {
//				_ranks[i] = currMax;
//			} else {
//				currMax = _ranks[i];
//			}
//		}
//
//		// Right to left sweep
//		currMax = 1;
//		for (size_t i = 1; i < numComponents; ++i) {
//			currMax *= _dimensions[numComponents-i];
//			if (isOperator) { currMax *= _dimensions[2*numComponents-i]; }
//
//			if (currMax < _ranks[numComponents-i-1]) {
//				_ranks[numComponents-i-1] = currMax;
//			} else {
//				currMax = _ranks[numComponents-i-1];
//			}
//		}
//
//		return _ranks;
//	}




	template<bool isOperator>
	void HTNetwork<isOperator>::use_dense_representations() {
		for (size_t i = 0; i < get_number_of_components(); ++i) {
			component(i).use_dense_representation();
		}
	}

	template<bool isOperator>
	Tensor& HTNetwork<isOperator>::component(const size_t _idx) {
		REQUIRE(_idx >= 0 || _idx < get_number_of_components(), "Illegal index " << _idx <<" in HTNetwork::component, as there are only " << get_number_of_components() << " components.");
		return *nodes[degree() == 0 ? 0 : _idx].tensorObject; //TODO check degree == 0
	}


	template<bool isOperator>
	const Tensor& HTNetwork<isOperator>::get_component(const size_t _idx) const {
		REQUIRE(_idx >= 0 || _idx < get_number_of_components(), "Illegal index " << _idx <<" in HTNetwork::get_component.");
		return *nodes[degree() == 0 ? 0 : _idx].tensorObject; //TODO check degree == 0
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::set_component(const size_t _idx, Tensor _T) {
		if(degree() == 0) {
			REQUIRE(_idx == 0, "Illegal index " << _idx <<" in HTNetwork::set_component");
			REQUIRE(_T.degree() == 0, "Component of degree zero HTNetwork must have degree zero. Given: " << _T.degree());
			*nodes[0].tensorObject = std::move(_T);
		} else {
			const bool isleave = _idx >= get_number_of_components() - degree()/N;
			REQUIRE(_idx < get_number_of_components(), "Illegal index " << _idx <<" in TTNetwork::set_component");
			REQUIRE(_idx >= 0, "Illegal index " << _idx <<" in TTNetwork::set_component");
			REQUIRE(isleave ? _T.degree() == N+1 : _T.degree() == 3, "Component " << _idx << " has degree: " << _T.degree() << " leave? " << isleave);

			size_t order = _T.degree();
			//size_t numberOfDummyComponents = static_cast<size_t>(numberOfComponents) - 2 * degree()/N + 1; //TODO check this
			TensorNode& currNode = nodes[_idx];

			*currNode.tensorObject = std::move(_T);
			for (size_t i = 0; i < order; ++i) {
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
	void HTNetwork<isOperator>::move_core(const size_t _position, const bool _keepRank) {
		const size_t numComponents = get_number_of_components();
		REQUIRE(_position < numComponents || (_position == 0 && degree() == 0), "Illegal core-position " << _position << " chosen for HTNetwork with " << numComponents << " components");
		require_correct_format();

		if(numComponents == 1) {
			REQUIRE(_position == 0, "If there is only one component it needs to be the core component");
			canonicalized = true;
			corePosition = _position;
			return;
		}

		if (!canonicalized){//canonicalize to 0
			for (size_t n = numComponents - 1; n > 0; --n) {
			  transfer_core(n, (n + 1) / 2 - 1, !_keepRank);
			  corePosition = 0;
			}
		}

		std::vector<size_t> path = get_path(corePosition, _position);

		while (path.size() > 1){
			size_t start = path.back();
			path.pop_back();
			size_t end = path.back();
		  transfer_core(start, end, !_keepRank);
		}

// TODO whz is this here??
//		while (exceeds_maximal_ranks()) {
//			// Move left from given CorePosition
//			for (size_t n = _position; n > 0; --n) {
//				transfer_core(n+1, n, !_keepRank);
//			}
//
//			// Move to the most right
//			for (size_t n = 0; n+1 < numComponents; ++n) {
//				transfer_core(n+1, n+2, !_keepRank);
//			}
//
//			// Move back left to given CorePosition
//			for (size_t n = numComponents; n > _position+1; --n) {
//				transfer_core(n, n-1, !_keepRank);
//			}
//		}

		canonicalized = true;
		corePosition = _position;
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::canonicalize_root() {
		move_core(0);
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::round(const std::vector<size_t>& _maxRanks, const double _eps) {
		require_correct_format();
		const size_t numOfLeaves = degree()/N;
		const size_t numIntComp = numOfLeaves - 1;
		const size_t numComponents = numIntComp + numOfLeaves;		REQUIRE(_eps < 1, "_eps must be smaller than one. " << _eps << " was given.");
		REQUIRE(_maxRanks.size()+1 == numComponents || (_maxRanks.empty() && numComponents == 0) ,"There must be exactly degree/N-1 maxRanks. Here " << _maxRanks.size() << " instead of " << numComponents-1 << " are given.");

		REQUIRE(!misc::contains(_maxRanks, size_t(0)), "Trying to round a HTTensor to rank 0 is not possible.");

		const bool initialCanonicalization = canonicalized;
		const size_t initialCorePosition = corePosition;

		canonicalize_root();

		for (size_t n = numComponents - 1; n > 0; --n) {
			round_edge(n, (n + 1) / 2 - 1, _maxRanks[n], _eps, 0.0);
		}

		assume_core_position(0);

		if(initialCanonicalization) {
			move_core(initialCorePosition);
		}
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::round(const size_t _maxRank) {
		round(std::vector<size_t>(num_ranks(), _maxRank), EPSILON);
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::round(const int _maxRank) {
		REQUIRE( _maxRank > 0, "MaxRank must be positive");
		round(size_t(_maxRank));
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::round(const value_t _eps) {
		round(std::vector<size_t>(num_ranks(), std::numeric_limits<size_t>::max()), _eps);
	}
//
//
	template<bool isOperator>
	void HTNetwork<isOperator>::soft_threshold(const std::vector<double> &_taus, const bool  /*_preventZero*/) {
		REQUIRE(_taus.size()+1 == get_number_of_components() || (_taus.empty() && get_number_of_components() == 0), "There must be exactly " << get_number_of_components() << " taus. Here " << _taus.size() << " instead of " << get_number_of_components()-1 << " are given.");
		require_correct_format();

		const bool initialCanonicalization = canonicalized;
		const size_t initialCorePosition = corePosition;
		const size_t numComp = get_number_of_components();

		canonicalize_root();

		for(size_t i = 0; i+1 < numComp; ++i) {
			round_edge(numComp-i, numComp-i-1, std::numeric_limits<size_t>::max(), 0.0, _taus[i]);
		}

		assume_core_position(0);

		if(initialCanonicalization) {
			move_core(initialCorePosition);
		}
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::soft_threshold(const double _tau, const bool _preventZero) {
		soft_threshold(std::vector<double>(num_ranks(), _tau), _preventZero);
	}
//
//
	template<bool isOperator>
	std::vector<size_t> HTNetwork<isOperator>::ranks() const {
		std::vector<size_t> res;
		res.reserve(num_ranks());
		const size_t numIntComp = degree()/N - 1;
		for (size_t n = 0; n < numIntComp; ++n) {
			res.push_back(nodes[n].neighbors.end()[-2].dimension);
			res.push_back(nodes[n].neighbors.back().dimension);
		}
		return res;
	}


	template<bool isOperator>
	size_t HTNetwork<isOperator>::rank(const size_t _i) const {
		REQUIRE(_i < get_number_of_components(), "Requested illegal rank " << _i);
		return nodes[_i].neighbors.back().dimension;
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::assume_core_position(const size_t _pos) {
		REQUIRE(_pos < degree()/N || (degree() == 0 && _pos == 0), "Invalid core position.");
		corePosition = _pos;
		canonicalized = true;
	}


	template<bool isOperator>
	TensorNetwork* HTNetwork<isOperator>::get_copy() const {
		return new HTNetwork(*this);
	}


//
	template<bool isOperator>
	value_t HTNetwork<isOperator>::frob_norm() const {
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
	HTNetwork<isOperator>& HTNetwork<isOperator>::operator+=(const HTNetwork<isOperator>& _other) {
		REQUIRE(dimensions == _other.dimensions, "The dimensions in HT sum must coincide. Given " << dimensions << " vs " << _other.dimensions);
		require_correct_format();
		size_t numComp = get_number_of_components();
		const size_t numLeaves = degree()/N;
		const size_t numInternalComponents = numComp - numLeaves;

		const bool initialCanonicalization = canonicalized;
		const size_t initialCorePosition = corePosition;

		if (numComp <= 1) {
			component(0) += _other.get_component(0);
			return *this;
		}

		XERUS_PA_START;
		for(size_t position = 0; position < numComp; ++position) {
			bool isLeaf = position >= numInternalComponents;
//			bool hasDummyLeftChild = false;
//			bool hasDummyRightChild = false;
//			if(!isLeaf){
//				hasDummyLeftChild = get_left_child_component(position) >= numComp;
//				hasDummyRightChild = get_right_child_component(position) >= numComp;
//			}
			// Get current components
			const Tensor& myComponent = get_component(position);
			const Tensor& otherComponent = _other.get_component(position);
			const Tensor::Representation newRep = myComponent.is_sparse() && otherComponent.is_sparse() ? Tensor::Representation::Sparse : Tensor::Representation::Dense;


			// Create a Tensor for the result
			std::vector<size_t> nxtDimensions;
			nxtDimensions.emplace_back(position == 0 ? 1 : myComponent.dimensions.front()+otherComponent.dimensions.front());

			nxtDimensions.emplace_back(isLeaf ? myComponent.dimensions[1] : myComponent.dimensions[1] + otherComponent.dimensions[1]);
			if (isOperator && isLeaf) { nxtDimensions.emplace_back(myComponent.dimensions[2]); }
			if (!isLeaf) {nxtDimensions.emplace_back(myComponent.dimensions[2] + otherComponent.dimensions[2]);}

			std::unique_ptr<Tensor> newComponent(new Tensor(std::move(nxtDimensions), newRep));

			newComponent->offset_add(myComponent, (!isOperator && isLeaf) ? std::vector<size_t>({0,0}) : std::vector<size_t>({0,0,0}));

			const size_t parentOffset = position == 0 ? 0 : myComponent.dimensions.front();
			const size_t child1Offset = isLeaf  ? 0 : myComponent.dimensions[1];
			const size_t child2Offset = isLeaf  ? 0 : myComponent.dimensions[2];
			newComponent->offset_add(otherComponent, !isOperator && isLeaf ?  std::vector<size_t>({parentOffset,child1Offset})  : std::vector<size_t>({parentOffset,child1Offset,child2Offset}) );


			set_component(position, std::move(*newComponent));
		}

		XERUS_PA_END("ADD/SUB", "HTNetwork ADD/SUB", std::string("Dims:")+misc::to_string(dimensions)+" Ranks: "+misc::to_string(ranks()));

		if(initialCanonicalization) {
			move_core(initialCorePosition);
		}

		return *this;
	}


	template<bool isOperator>
	HTNetwork<isOperator>& HTNetwork<isOperator>::operator-=(const HTNetwork<isOperator>& _other) {
		operator*=(-1.0);
		operator+=(_other);
		operator*=(-1.0);
		return *this;
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::operator*=(const value_t _factor) {
		REQUIRE(!nodes.empty(), "There must not be a HTNetwork without any node");
		if(canonicalized) {
			component(corePosition) *= _factor;
		} else {
			component(0) *= _factor;
		}
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::operator/=(const value_t _divisor) {
		operator*=(1/_divisor);
	}



	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */



	template<>
	bool HTNetwork<false>::specialized_contraction_f(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& /*unused*/, internal::IndexedTensorReadOnly<TensorNetwork>&& /*unused*/, internal::IndexedTensorReadOnly<TensorNetwork>&& /*unused*/) {
		// Only HTOperators construct stacks, so no specialized contractions for HTTensors
		return false;
	}

	template<>
	bool HTNetwork<true>::specialized_contraction_f(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& _out, internal::IndexedTensorReadOnly<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		_me.assign_indices();
		_other.assign_indices();

		const HTNetwork* const meHT = dynamic_cast<const HTNetwork*>(_me.tensorObjectReadOnly);
		const internal::HTStack<true>* const meHTStack = dynamic_cast<const internal::HTStack<true>*>(_me.tensorObjectReadOnly);
		INTERNAL_CHECK(meHT || meHTStack, "Internal Error.");

		const HTTensor* const otherHT = dynamic_cast<const HTTensor*>(_other.tensorObjectReadOnly);
		const internal::HTStack<false>* const otherHTStack = dynamic_cast<const internal::HTStack<false>*>(_other.tensorObjectReadOnly);
		const HTOperator* const otherHTO = dynamic_cast<const HTOperator*>(_other.tensorObjectReadOnly);
		const internal::HTStack<true>* const otherHTOStack = dynamic_cast<const internal::HTStack<true>*>(_other.tensorObjectReadOnly);

		if ((otherHT == nullptr)  && (otherHTO == nullptr) && (otherHTOStack == nullptr) && (otherHTStack == nullptr)) {
			return false;
		}

		bool cannoAtTheEnd = false;
		size_t coreAtTheEnd = 0;
		if (meHT != nullptr) {
			cannoAtTheEnd = meHT->canonicalized;
			coreAtTheEnd = meHT->corePosition;
		} else {
			cannoAtTheEnd = meHTStack->cannonicalization_required;
			coreAtTheEnd = meHTStack->futureCorePosition;
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

		if ((otherHT != nullptr) || (otherHTStack != nullptr)) {
			// ensure fitting indices
			if (std::equal(_me.indices.begin(), midIndexItr, _other.indices.begin()) || std::equal(midIndexItr, _me.indices.end(), _other.indices.begin())) {
				_out.reset(new internal::IndexedTensorMoveable<TensorNetwork>(new internal::HTStack<false>(cannoAtTheEnd, coreAtTheEnd), _me.indices));
				*_out->tensorObject = *_me.tensorObjectReadOnly;
				TensorNetwork::add_network_to_network(std::move(*_out), std::move(_other));
				return true;
			}
				return false;

		}
		else { // other is operator or operator stack
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
			// or indices in fitting order to contract the HTOs
			if (   std::equal(_me.indices.begin(), midIndexItr, _other.indices.begin())
				|| std::equal(midIndexItr, _me.indices.end(), _other.indices.begin())
				|| std::equal(_me.indices.begin(), midIndexItr, otherMidIndexItr)
				|| std::equal(midIndexItr, _me.indices.end(), otherMidIndexItr))
			{
				_out.reset(new internal::IndexedTensorMoveable<TensorNetwork>(new internal::HTStack<true>(cannoAtTheEnd, coreAtTheEnd), _me.indices));
				*_out->tensorObject = *_me.tensorObjectReadOnly;
				TensorNetwork::add_network_to_network(std::move(*_out), std::move(_other));
				return true;
			}
				return false;

		}
		return false;
	}


	template<bool isOperator>
	void transpose_if_operator(HTNetwork<isOperator>& _htNetwork);

	template<>
	void transpose_if_operator<false>(HTNetwork<false>&  /*_ttNetwork*/) {}

	template<>
	void transpose_if_operator<true>(HTNetwork<true>& _htNetwork) {
		_htNetwork.transpose();
	}

	template<bool isOperator>
	bool HTNetwork<isOperator>::specialized_sum_f(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& _out, internal::IndexedTensorReadOnly<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		_me.assign_indices();
		_other.assign_indices();

		// If the other is not a HT tensor (or stack) fall back to default summation (i.e. return false)
		const HTNetwork* otherHT = dynamic_cast<const HTNetwork*>( _other.tensorObjectReadOnly);
		const internal::HTStack<isOperator>* otherHTStack = dynamic_cast<const internal::HTStack<isOperator>*>( _other.tensorObjectReadOnly);
		if (!otherHT && !otherHTStack) { return false; }

		bool transposeRHS;
		if(_me.indices == _other.indices) { // Everything is easy.
			REQUIRE(_me.tensorObjectReadOnly->dimensions == _other.tensorObjectReadOnly->dimensions, "HT sum requires both operants to share the same dimensions.");
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

		// Check whether we are a HTStack
		std::unique_ptr<HTNetwork<isOperator>> meStorage;
		const HTNetwork* usedMe;

		internal::IndexedTensorMoveable<TensorNetwork>* const moveMe = dynamic_cast<internal::IndexedTensorMoveable<TensorNetwork>*>(&_me);
		internal::HTStack<isOperator>* stackMe;
		if(moveMe && (stackMe = dynamic_cast<internal::HTStack<isOperator>*>(moveMe->tensorObject))) {
			meStorage.reset(new HTNetwork());
			usedMe = meStorage.get();
			*meStorage = HTNetwork(*stackMe);
			INTERNAL_CHECK(usedMe->dimensions == stackMe->dimensions, "Ie " << stackMe->dimensions << " vs "  << usedMe->dimensions);
		} else { // I am normal
			INTERNAL_CHECK(dynamic_cast<const HTNetwork<isOperator>*>(_me.tensorObjectReadOnly),"Non-moveable HTStack (or other error) detected.");
			usedMe = static_cast<const HTNetwork<isOperator>*>(_me.tensorObjectReadOnly);
		}
		const HTNetwork& htMe = *usedMe;


		// Check whether the other is a HTStack
		HTNetwork htOther;

		internal::IndexedTensorMoveable<TensorNetwork>* const moveOther = dynamic_cast<internal::IndexedTensorMoveable<TensorNetwork>*>(&_other);
		internal::HTStack<isOperator>* stackOther;
		if(moveOther && (stackOther = dynamic_cast<internal::HTStack<isOperator>*>(moveOther->tensorObject))) {
			htOther = HTNetwork(*stackOther);
			INTERNAL_CHECK(htOther.dimensions == stackOther->dimensions, "Ie");
		} else { // Other is normal
			INTERNAL_CHECK(dynamic_cast<const HTNetwork<isOperator>*>(_other.tensorObjectReadOnly),"Non-moveable HTStack (or other error) detected.");
			htOther = *static_cast<const HTNetwork<isOperator>*>(_other.tensorObjectReadOnly);

		}

		if(transposeRHS) {
			transpose_if_operator(htOther);
		}

		_out.reset(new internal::IndexedTensorMoveable<TensorNetwork>( new HTNetwork(htMe), _me.indices));

		*static_cast<HTNetwork*>(_out->tensorObject) += htOther;
		return true;
	}


	template<bool isOperator>
	void HTNetwork<isOperator>::specialized_evaluation(internal::IndexedTensorWritable<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		INTERNAL_CHECK(_me.tensorObject == this, "Internal Error.");

		_me.assign_indices(_other.degree());
		_other.assign_indices();

		//const size_t numIntComp = static_cast<size_t>(0.5+std::pow(2.,std::ceil(std::log2(static_cast<double>(_other.degree()/N ))))) - 1;
		const size_t numOfLeaves = _other.degree()/N;
		//const size_t numComponents = numIntComp + numOfLeaves;

		HTNetwork& meHTN = static_cast<HTNetwork&>(*_me.tensorObject);

		// First check whether the other is a HTNetwork as well, otherwise we can skip to fallback
		const HTNetwork* const otherHTN = dynamic_cast<const HTNetwork*>(_other.tensorObjectReadOnly);
		const internal::HTStack<isOperator>* const otherHTStack = dynamic_cast<const internal::HTStack<isOperator>*>(_other.tensorObjectReadOnly);
		internal::IndexedTensorMoveable<TensorNetwork> *movOther = dynamic_cast<internal::IndexedTensorMoveable<TensorNetwork> *>(&_other);
		if(otherHTN  || otherHTStack) {

			if (otherHTStack) {
				INTERNAL_CHECK(movOther, "Not moveable HTStack encountered...");
				internal::HTStack<isOperator>::contract_stack(std::move(*movOther));
			}

			// Check whether the index order coincides
			if (_me.indices == _other.indices) {
				if (otherHTN) {
					meHTN = *otherHTN;
				} else {
					_me.tensorObject->operator=(*_other.tensorObjectReadOnly);
					meHTN.canonicalized = false;
					if (otherHTStack->cannonicalization_required) {
						meHTN.move_core(otherHTStack->futureCorePosition);
					}
				}
				return;
			}

			// For HTOperators also check whether the index order is transposed
			if (isOperator) {
				bool transposed = false;

				auto midIndexItr = _me.indices.begin();
				size_t spanSum = 0;
				while (spanSum < numOfLeaves) {
					INTERNAL_CHECK(midIndexItr != _me.indices.end(), "Internal Error.");
					spanSum += midIndexItr->span;
					++midIndexItr;
				}
				if (spanSum == numOfLeaves) {
					// Transposition possible on my end
					auto otherMidIndexItr = _other.indices.begin();
					spanSum = 0;
					while (spanSum < numOfLeaves) {
						INTERNAL_CHECK(otherMidIndexItr != _other.indices.end(), "Internal Error.");
						spanSum += otherMidIndexItr->span;
						++otherMidIndexItr;
					}
					if (spanSum == numOfLeaves) {
						// Other tensor also transposable
						transposed = (std::equal(_me.indices.begin(), midIndexItr, otherMidIndexItr))
						&& (std::equal(midIndexItr, _me.indices.end(), _other.indices.begin()));
					}
				}
				if (transposed) {
					if (otherHTN) {
						meHTN = *otherHTN;
					} else {
						_me.tensorObject->operator=(*_other.tensorObjectReadOnly);
						meHTN.canonicalized = false;
						if (otherHTStack->cannonicalization_required) {
							meHTN.move_core(otherHTStack->futureCorePosition);
						}
					}
					require_correct_format();
					dynamic_cast<HTOperator*>(_me.tensorObject)->transpose(); // NOTE will never be called if !isOperator
					return;
				}
			}
		}
		// Use Tensor fallback
		if (_other.tensorObjectReadOnly->nodes.size() > 1) {
			LOG_ONCE(warning, "Assigning a general tensor network to HTOperator not yet implemented. casting to fullTensor first");
		}
		Tensor otherFull(*_other.tensorObjectReadOnly);
		Tensor otherReordered;
		otherReordered(_me.indices) = otherFull(_other.indices);

		// Cast to HTNetwork
		*_me.tensorObject = HTNetwork(std::move(otherReordered));
	}


	// Explicit instantiation of the two template parameters that will be implemented in the xerus library
	template class HTNetwork<false>;
	template class HTNetwork<true>;



	template<bool isOperator>
	HTNetwork<isOperator> operator+(HTNetwork<isOperator> _lhs, const HTNetwork<isOperator>& _rhs) {
		_lhs += _rhs; // NOTE pass-by-value!
		return _lhs;
	}


	template<bool isOperator>
	HTNetwork<isOperator> operator-(HTNetwork<isOperator> _lhs, const HTNetwork<isOperator>& _rhs) {
		_lhs -= _rhs; // NOTE pass-by-value!
		return _lhs;
	}


	template<bool isOperator>
	HTNetwork<isOperator> operator*(HTNetwork<isOperator> _network, const value_t _factor) {
		_network *= _factor; // NOTE pass-by-value!
		return _network;
	}


	template<bool isOperator>
	HTNetwork<isOperator> operator*(const value_t _factor, HTNetwork<isOperator> _network) {
		_network *= _factor; // NOTE pass-by-value!
		return _network;
	}


	template<bool isOperator>
	HTNetwork<isOperator> operator/(HTNetwork<isOperator> _network, const value_t _divisor) {
		_network /= _divisor; // NOTE pass-by-value!
		return _network;
	}

	// Explicit instantiation for both types
	template HTNetwork<false> operator+(HTNetwork<false> _lhs, const HTNetwork<false>& _rhs);
	template HTNetwork<true> operator+(HTNetwork<true> _lhs, const HTNetwork<true>& _rhs);
	template HTNetwork<false> operator-(HTNetwork<false> _lhs, const HTNetwork<false>& _rhs);
	template HTNetwork<true> operator-(HTNetwork<true> _lhs, const HTNetwork<true>& _rhs);
	template HTNetwork<false> operator*(HTNetwork<false> _network, const value_t _factor);
	template HTNetwork<true> operator*(HTNetwork<true> _network, const value_t _factor);
	template HTNetwork<false> operator*(const value_t _factor, HTNetwork<false> _network);
	template HTNetwork<true> operator*(const value_t _factor, HTNetwork<true> _network);
	template HTNetwork<false> operator/(HTNetwork<false> _network, const value_t _divisor);
	template HTNetwork<true> operator/(HTNetwork<true> _network, const value_t _divisor);


	namespace misc {

		template<bool isOperator>
		void stream_writer(std::ostream& _stream, const HTNetwork<isOperator> &_obj, misc::FileFormat _format) {
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
		template void stream_writer(std::ostream& _stream, const HTNetwork<true> &_obj, misc::FileFormat _format);
		template void stream_writer(std::ostream& _stream, const HTNetwork<false> &_obj, misc::FileFormat _format);


		template<bool isOperator>
		void stream_reader(std::istream& _stream, HTNetwork<isOperator> &_obj, const misc::FileFormat _format) {
			IF_CHECK( size_t ver = ) read_from_stream<size_t>(_stream, _format);
			REQUIRE(ver == 1, "Unknown stream version to open (" << ver << ")");

			// load TN specific data
			read_from_stream<bool>(_stream, _obj.canonicalized, _format);
            read_from_stream<size_t>(_stream, _obj.corePosition, _format);


			// load rest of TN
			read_from_stream<TensorNetwork>(_stream, _obj, _format);

			_obj.require_correct_format();
		}
		template void stream_reader(std::istream& _stream, HTNetwork<true> &_obj, const misc::FileFormat _format);
		template void stream_reader(std::istream& _stream, HTNetwork<false> &_obj, const misc::FileFormat _format);
	} // namespace misc
	
} // namespace xerus

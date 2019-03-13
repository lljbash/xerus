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
 * @brief Common functions for the Unittests.
 */

#include "../../include/xerus.h"
#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

size_t tt_dofs(const std::vector<size_t>& _dimensions, const std::vector<size_t>& _ranks) {
	REQUIRE(_dimensions.size() == _ranks.size()+1, "Inconsitend order.");
	
	const size_t order = _dimensions.size();
	size_t dofs = 0;
	
	dofs += _dimensions[0]*_ranks[0];
	
	for(size_t mu = 1; mu+1 < order; ++mu) {
		dofs += _ranks[mu-1]*_dimensions[mu]*_ranks[mu];
	}
	
	dofs += _ranks[order-1]*_dimensions[order-1];
	
	for(size_t mu = 0; mu+2 < order; ++mu) {
		dofs -= misc::sqr(_ranks[mu]);
	}
	
	return dofs;
	
}

std::vector<size_t> random_dimensions(const size_t _order, const size_t _minN = 1, const size_t _maxN = 10) {
	std::uniform_int_distribution<size_t> dimDist(_minN, _maxN);
	std::vector<size_t> dimensions;
	for(size_t d = 0; d < _order; ++d) {
		dimensions.push_back(dimDist(misc::randomEngine));
	}
	return dimensions;
}

std::vector<size_t> random_tt_ranks(const std::vector<size_t>& _dimensions, const size_t _minN = 1, const size_t _maxN = 10) {
	std::uniform_int_distribution<size_t> dimDist(_minN, _maxN);
	std::vector<size_t> ranks;
	for(size_t d = 1; d < _dimensions.size(); ++d) {
		ranks.push_back(dimDist(misc::randomEngine));
	}
	return TTNetwork<false>::reduce_to_maximal_ranks(ranks, _dimensions);
}

std::vector<size_t> random_low_tt_ranks(const std::vector<size_t>& _dimensions, const size_t _minN = 1, size_t _maxN = 10) {
	std::vector<size_t> ranks;
	do {
		std::uniform_int_distribution<size_t> dimDist(_minN, _maxN);
		
		ranks.clear();
		for(size_t d = 1; d < _dimensions.size(); ++d) {
			ranks.push_back(dimDist(misc::randomEngine));
		}
		
		ranks = TTNetwork<false>::reduce_to_maximal_ranks(ranks, _dimensions);
		
		--_maxN;
	} while (_maxN > 2 && tt_dofs(_dimensions, ranks) > misc::product(_dimensions)/10);
	
	return ranks;
}

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
 * @brief Implementation of the ALS variants.
 */

#include <xerus/algorithms/randomSVD.h>
#include <xerus/basic.h>

#include "xerus/misc/internal.h"


namespace xerus {

TTTensor randomTTSVD(const Tensor& _x, const std::vector<size_t>& _ranks) {
    const size_t d = _x.degree();
	
	REQUIRE(d == _ranks.size()+1, "Inconsitend degree vs number of provided ranks.");
	
    TTTensor u(d);
    Tensor a;
	Tensor b = _x;
    
    for(size_t j = d-1; j > 0; --j) {
		const size_t contractSize = misc::product(b.dimensions, 0, j);
        const size_t staySize = misc::product(b.dimensions, j, b.dimensions.size());
        const size_t s = std::min(_ranks[j-1], std::min(contractSize, staySize)+1);
        
        if(b.is_sparse()) {
            std::map<size_t, std::vector<value_t>> usedG;
			
			std::vector<size_t> outDims({s});
			outDims.insert(outDims.end(), b.dimensions.cbegin()+j, b.dimensions.cend());
			a = Tensor(outDims, Tensor::Representation::Sparse, Tensor::Initialisation::Zero);
            const auto& data = b.get_sparse_data();
			
            for(const auto& entry : data) {
                const size_t pos = entry.first/staySize;
                const size_t outPos = entry.first%staySize;
                
                auto& gEntry = usedG[pos];
                if(gEntry.empty()) {
                    gEntry.reserve(s);
                    for(size_t k = 0; k < s; ++k) {
                        gEntry.push_back(misc::defaultNormalDistribution(misc::randomEngine));
                    }
                }
                
                for(size_t k = 0; k < s; ++k) {
                    a[k*staySize+outPos] += gEntry[k]*entry.second;
                }
            }
            
        } else {
            std::vector<size_t> gDims({s});
            gDims.insert(gDims.end(), b.dimensions.cbegin(), b.dimensions.cbegin()+j);
            const Tensor g = Tensor::random(gDims, misc::defaultNormalDistribution, misc::randomEngine);
            contract(a, g, false, b, false, j);
        }
        
        
        Tensor R, Q;
        calculate_cq(R, Q, a, 1); 
        
        
        if(j == d-1) {
            contract(b, b, false, Q, true, 1);
            Q.reinterpret_dimensions(Q.dimensions | std::vector<size_t>({1}));
            u.set_component(j, Q);
        } else {
            contract(b, b, false, Q, true, 2);
            u.set_component(j, Q); 
        }
    }
    
    b.reinterpret_dimensions(std::vector<size_t>({1}) | b.dimensions);
    u.set_component(0, b);
    
    return u;
}
    
    
TTTensor randomTTSVD(const Tensor& _x, const std::vector<size_t>& _ranks, const std::vector<size_t>& _oversampling) {
	REQUIRE(_ranks.size() == _oversampling.size(), "Inconsitend rank/oversampling sizes.");
	
    std::vector<size_t> sampRanks = _ranks;
	for(size_t i = 0; i < _ranks.size(); ++i) {
		sampRanks[i] += _oversampling[i];
	}
	
	auto ttX = randomTTSVD(_x, sampRanks);
    ttX.round(_ranks);
    
    return ttX;
}



} // namespace xerus


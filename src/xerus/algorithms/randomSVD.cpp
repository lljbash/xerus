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
 * @brief Implementation of the ALS variants.
 */

#include <xerus/algorithms/randomSVD.h>
#include <xerus/basic.h>

#include "xerus/misc/internal.h"


namespace xerus {
	
TTTensor randomTTSVD(const Tensor& _x, const std::vector<size_t>& _ranks) {
	REQUIRE(_x.degree() == _ranks.size()+1, "Inconsitend degree vs number of provided ranks.");
    std::normal_distribution<double> dist(0, 1);
    
    const size_t d = _x.degree();
    TTTensor u(d);
    Tensor b = _x;
    
    for(size_t j = d; j >= 2; --j) {
        const size_t s = _ranks[j-2];
        
        const std::vector<size_t> mixDims(b.dimensions.cbegin(), b.dimensions.cbegin()+(j-1));
        
        std::vector<size_t> outDims({s});
        outDims.insert(outDims.end(), b.dimensions.cbegin()+(j-1), b.dimensions.cend());
        
        Tensor a(outDims, Tensor::Representation::Sparse, Tensor::Initialisation::Zero);
        
        if(b.is_sparse()) {
            const size_t staySize = misc::product(b.dimensions, j-1, b.dimensions.size());
            
            std::map<size_t, std::vector<value_t>> usedG;
            
            const auto& data = b.get_sparse_data();
            for(const auto& entry : data) {
                const size_t pos = entry.first/staySize;
                const size_t outPos = entry.first%staySize;
                
                auto& gEntry = usedG[pos];
                if(gEntry.empty()) {
                    gEntry.reserve(s);
                    for(size_t k = 0; k < s; ++k) {
                        gEntry.push_back(dist(xerus::misc::randomEngine));
                    }
                }
                
                for(size_t k = 0; k < s; ++k) {
                    a[outPos+k*staySize] += gEntry[k]*entry.second;
                }
            }
            
        } else {
            std::vector<size_t> gDims({s});
            gDims.insert(gDims.end(), mixDims.cbegin(), mixDims.cend());
            const Tensor g = Tensor::random(gDims, dist, xerus::misc::randomEngine);
            contract(a, g, false, b, false, j-1);
        }
        
        
        Tensor R, Q;
        calculate_rq(R, Q, a, 1);
        
        
        if(j == d) {
            contract(b, b, false, Q, true, 1);
            Q.reinterpret_dimensions(Q.dimensions | std::vector<size_t>({1}));
            u.set_component(j-1, Q);
        } else {
            contract(b, b, false, Q, true, 2);
            u.set_component(j-1, Q); 
        }
    }
    
    b.reinterpret_dimensions(std::vector<size_t>({1}) | b.dimensions);
    u.set_component(0, b);
    
    u.round(_ranks);
    
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


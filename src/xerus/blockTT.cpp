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

#include <xerus/blockTT.h>
#include <xerus/ttNetwork.h>

#include <xerus/misc/check.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#include <xerus/basic.h>
#include <xerus/misc/basicArraySupport.h>

namespace xerus { namespace internal {
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
    
    BlockTT::BlockTT(const TTTensor& _tttensor, const size_t _blockPosition, const size_t _blockDim) : P(_blockDim), corePosition(_blockPosition), dimensions(_tttensor.dimensions) {
        REQUIRE(_tttensor.canonicalized && _tttensor.corePosition == _blockPosition, "Invalid core Position");
        _tttensor.require_correct_format();
        
        // Get components
        for(size_t i = 0; i < _tttensor.degree(); ++i) {
            components.push_back(_tttensor.get_component(i));
        }
        
        // Create block
        const Index left, right, ext, p;
        components[_blockPosition](left, ext, p, right) = components[_blockPosition](left, ext, right)*Tensor::ones({_blockDim})(p);
    }
    
    size_t BlockTT::degree() const {
        return components.size();
    }
    
    std::vector<size_t> BlockTT::ranks() const {
		std::vector<size_t> res;
		res.reserve(degree()-1);
		for (size_t n = 1; n < degree(); ++n) {
			res.push_back(components[n].dimensions[0]);
		}
		return res;
	}
	
	size_t BlockTT::rank(const size_t _idx) const {
        REQUIRE(_idx+1 < degree(), "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree() << " components.");
        return components[_idx+1].dimensions[0];
    }
    
    
    Tensor& BlockTT::component(const size_t _idx) {
        REQUIRE(_idx < degree(), "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree() << " components.");
        return components[_idx];
    }
			
    
    const Tensor& BlockTT::get_component(const size_t _idx) const {
        REQUIRE(_idx < degree(), "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree() << " components.");
        return components[_idx];
    }
    
    
    void BlockTT::set_component(const size_t _idx, const Tensor& _T) {
        REQUIRE(_idx == corePosition, "Illegal index " << _idx <<" as core position is " << corePosition << ".");
        REQUIRE(components[_idx].dimensions == _T.dimensions, "Invalid dimensions given: " << components[_idx].dimensions << " vs " << _T.dimensions);
        components[_idx] = _T;
    }
    
    
    Tensor BlockTT::get_core(const size_t _blockPos) const {
        REQUIRE(_blockPos < P, "IE");
        Tensor coreCmp = components[corePosition];
        coreCmp.fix_mode(2, _blockPos);
        return coreCmp;
    }
    
    
    Tensor BlockTT::get_average_core() const {
        const Index left, right, ext, p, r1, r2;
        Tensor coreCmp;
        coreCmp(left, ext, right) = (1.0/double(P))*components[corePosition](left, ext, p, right)*Tensor::ones({P})(p);
        return coreCmp;
    }
            
    
    value_t BlockTT::frob_norm() const {
        return components[corePosition].frob_norm();
    }
    
    
    size_t BlockTT::dofs() const {
        size_t numDofs = 0;
        
        for(const auto& comp : components) {
            numDofs += comp.size;
        }
        
        for(const auto r : ranks()) {
            numDofs -= r*r;
        }
        

        REQUIRE(corePosition == 0, "IE");
        numDofs -= components[0].size - misc::sqr(components[1].dimensions[0]);
        
        return numDofs;
    }
    
	
	void BlockTT::move_core(const size_t _position, const double _eps, const size_t _maxRank) {
        REQUIRE(_position < degree(), "IE");
        const Index left, right, ext, p, r1, r2;
        Tensor U, S, V;
        while(corePosition < _position) { // To right
            (U(left, ext, r1), S(r1, r2), V(r2, p, right)) = SVD(components[corePosition](left, ext, p, right), _maxRank, _eps);
            components[corePosition] = U;
            components[corePosition+1](left, ext, p, right) = S(left, r1)*V(r1, p, r2)*components[corePosition+1](r2, ext, right);
            corePosition++;
        }
        
        while(corePosition > _position) { // To left
            (U(left, p, r1), S(r1, r2), V(r2, ext, right)) = SVD(components[corePosition](left, ext, p, right), _maxRank, _eps);
            components[corePosition] = V;
            components[corePosition-1](left, ext, p, right) = components[corePosition-1](left, ext, r1)*U(r1, p, r2)*S(r2, right);
            corePosition--;
        }
    }
    
    
    
    void BlockTT::average_core() {
        const Index left, right, ext, p;
        Tensor coreCmp;
        coreCmp(left, ext, right) = (1.0/double(P))*components[corePosition](left, ext, p, right)*Tensor::ones({P})(p);
        components[corePosition](left, ext, p, right) = coreCmp(left, ext, right)*Tensor::ones({P})(p);
    }
	
}} // namespace xerus

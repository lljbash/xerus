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
 * @brief Implementation of the decomposition ALS.
 */

#include <xerus/algorithms/decompositionAls.h>
#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/indexedTensorList.h>
#include <xerus/tensorNetwork.h>
 
#include <xerus/indexedTensorMoveable.h>
#include <xerus/indexedTensor_tensor_factorisations.h>

namespace xerus {

	void decomposition_als(TTTensor& _x, const Tensor& _b, const double _eps, const size_t _numHalfSweeps) {
		Index k, iU, iX, iL, rU, rL;
		Tensor newXi;
		
		double lastResidual = frob_norm(Tensor(_x) - _b);
		
		for(size_t iteration = 0; iteration < _numHalfSweeps; iteration += 2) {
			// Move right
			for(size_t pos = 0; pos < _x.degree(); ++pos) { XERUS_REQUIRE_TEST;
				_x.move_core(pos);
				std::pair<TensorNetwork, TensorNetwork> split = _x.chop(pos);
				_x.component(pos)(rU, iX, rL) = split.first(iU&1, rU)*split.second(rL, iL&1)*_b(iU^pos, iX, iL&(pos+1));
			}
			
			// Move left
			if(iteration+1 < _numHalfSweeps) {
				for(size_t pos = _x.degree()-2; pos > 0; --pos) { XERUS_REQUIRE_TEST;
					_x.move_core(pos);
					std::pair<TensorNetwork, TensorNetwork> split = _x.chop(pos);
					_x.component(pos)(rU, iX, rL) = split.first(iU&1, rU)*split.second(rL, iL&1)*_b(iU^pos, iX, iL&(pos+1));
				}
			}
			
			double residual = frob_norm(Tensor(_x) - _b);
			if(residual < EPSILON || (lastResidual-residual)/residual < _eps) { return; }
			lastResidual = residual;
		}
	}
} // namespace xerus

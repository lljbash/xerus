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
 * @brief Implementation of the examples of low-rank TT tensors defined in the respective header file.
 */

#include <xerus/examples/specificLowRankTensors.h>
#include <xerus/misc/internal.h>

namespace xerus { namespace examples {
	
	TTTensor peaking_diagonals(size_t _degree, size_t _n, value_t _alpha) {
		REQUIRE(_degree >= 2, "");
		REQUIRE(_n>=2, "");
		TTTensor e1(Tensor({_n}, [](){return 1.0;}));
		TTTensor cross(Tensor({_n,_n}, [&](const std::vector<size_t> &idx){
			return 1.0/(double(idx[0]>idx[1]?idx[0]-idx[1]:idx[1]-idx[0]) + _alpha) + 1.0/(double(idx[0])+_alpha) + 1.0/(double(idx[1])+_alpha);
		}));
		
		TTTensor result(cross);
		TTTensor buffer(e1);
		while (result.degree() < _degree) {
			result = dyadic_product(result, e1);
			TTTensor tmp = dyadic_product(buffer, cross);
			result += tmp;
			result.round(0.0);
			buffer = dyadic_product(buffer, e1);
		}
		return result;
	}
	
} // namespace examples
} // namespace xerus


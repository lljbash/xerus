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


#include <xerus.h>

#include "../../include/xerus/test/test.h"

using namespace xerus;

static misc::UnitTest decomp_als("ALS", "decomposition_als", [](){
	xerus::Index i,j,k;
	
	const size_t d = 14;
	const size_t n = 2;

	const std::vector<size_t> stateDims(d, n);
	
    xerus::TTTensor TTB = xerus::TTTensor::random(stateDims, 4);
	Tensor B(TTB);
	
	xerus::TTTensor X = xerus::TTTensor::random(stateDims, 4);
	
	xerus::decomposition_als(X, B);
	
	TEST(frob_norm(X(i&0)-B(i&0)) < 1e-8);
});

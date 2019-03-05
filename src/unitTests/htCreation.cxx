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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
using namespace xerus;


static misc::UnitTest ht_identity("HT", "identity", [](){
	Tensor A = Tensor::identity({2,3,2,2,3,2});
	HTOperator htA = HTOperator::identity({2,3,2,2,3,2});

	Tensor B = Tensor::identity({2,3,2,3,4,2,3,2,3,4});
	HTOperator htB = HTOperator::identity({2,3,2,3,4,2,3,2,3,4});

	TEST(approx_equal(A, Tensor(htA), 1e-14));
	TEST(approx_equal(B, Tensor(htB), 1e-14));
});

static misc::UnitTest ht_ones("HT", "ones", [](){
	Tensor A = Tensor::ones({2,3,2,2,3,2});
	HTOperator htA = HTOperator::ones({2,3,2,2,3,2});

	Tensor B = Tensor::ones({2,3,2,2,3,2});
	HTTensor htB = HTTensor::ones({2,3,2,2,3,2});

	TEST(approx_equal(A, Tensor(htA), 1e-14));
	TEST(approx_equal(B, Tensor(htB), 1e-14));
});

static misc::UnitTest ht_dirac("HT", "dirac", [](){
	Tensor B = Tensor::dirac({2,3,2,2,3,2},5);
	HTTensor htB = HTTensor::dirac({2,3,2,2,3,2},5);

	TEST(approx_equal(B, Tensor(htB), 1e-14));

	Tensor A = Tensor::dirac({2,3,2,2,3,2},5);
	HTOperator htA = HTOperator::dirac({2,3,2,2,3,2},5);

	TEST(approx_equal(A, Tensor(htA), 1e-14));

});

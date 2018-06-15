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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
using namespace xerus;

static misc::UnitTest tensor_factors("Tensor", "Factors", [](){
    Tensor A = Tensor::random({2,7,5,5,2,7});
    Tensor B = Tensor::random({2,7,5,5,2,7});
    Tensor A3 = 3*A;
    Tensor B7 = 7*B;
	TEST(!A.has_factor());
	TEST(!B.has_factor());
	TEST(A3.has_factor());
	TEST(B7.has_factor());
	
    Tensor Q;
    Tensor R;
    Tensor res1;
    Tensor res2;
    Tensor res3;
    Tensor res4;
    
    Index i, j, k, l, m, n, o, p, r, s;
    
    (res1(i,j,k,o), res2(o,p), res3(p,l,m,n)) = SVD(3*A(i,j,k,l,m,n));
    res4(i,j,k,l,m,n) = 3.7*res1(i,j,k,o)*(res2(o,p)/3.7)*res3(p,l,m,n);
    TEST(approx_equal(res4, A3, 1e-11));
    
    (Q(i,j,k,l), R(l,m,n,r)) = QR(B7(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = (Q(i,j,k,o)/12.5)*(12.5*R(o,m,n,r)/7);
    TEST(approx_equal(res4, B, 1e-12));
});

static misc::UnitTest tensor_value_t("Tensor", "value_t_Product", [](){
    Tensor A({4,2,2,7});
    Tensor B;
    Tensor C;
    Tensor D;
    for(size_t i = 0; i < A.size; ++i) {
        A[i] = 73;
    }
    
    Index j;
    
    B(j&0) = A(j&0)*2.0;
    C(j&0) = 3*A(j&0);
    D(j&0) = A(j&0)/73.0;
    A(j&0) = A(j&0)/2;
    
    TEST(approx_entrywise_equal(B, std::vector<value_t>(A.size, 146)));
    TEST(approx_entrywise_equal(C, std::vector<value_t>(A.size, 219)));
    TEST(approx_entrywise_equal(D, std::vector<value_t>(A.size, 1)));
    TEST(approx_entrywise_equal(A, std::vector<value_t>(A.size, 36.5)));
});

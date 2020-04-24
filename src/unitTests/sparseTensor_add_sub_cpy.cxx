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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
using namespace xerus;

static misc::UnitTest sparse_sum2("SparseTensor", "sum_matrix_2x2", [](){
    Tensor res({2,2}, Tensor::Representation::Sparse);
    Tensor B({2,2}, Tensor::Representation::Sparse);
    Tensor C({2,2}, Tensor::Representation::Sparse);

    Index i, j;
    
    B[{0,0}]=1;
    B[{0,1}]=2;
    B[{1,0}]=3;
    B[{1,1}]=4;
    
    C[{0,0}]=5;
    C[{0,1}]=6;
    C[{1,0}]=7;
    C[{1,1}]=8;
    
	B.use_sparse_representation();
	C.use_sparse_representation();
	
    res(i,j) = B(i,j) + C(i,j);
    TEST(approx_entrywise_equal(res, {6,8,10,12}));
    res(i,j) = B(i,j) + C(j,i);
    TEST(approx_entrywise_equal(res, {6,9,9,12}));
});
 
static misc::UnitTest sparse_sum_eq("SparseTensor", "sum_lhs_equals_rhs", [](){
    Tensor B({2,2}, Tensor::Representation::Sparse);
    Tensor C({2,2}, Tensor::Representation::Sparse);

    Index i, j;
    
    B[{0,0}]=1;
    B[{0,1}]=2;
    B[{1,0}]=3;
    B[{1,1}]=4;
    
    C[{0,0}]=5;
    C[{0,1}]=6;
    C[{1,0}]=7;
    C[{1,1}]=8;
    
	B.use_sparse_representation();
	C.use_sparse_representation();
    B(i,j) = B(i,j) + C(i,j);
	TEST(approx_entrywise_equal(B, {6,8,10,12}));
    B(i,j) = B(i,j) + B(j,i);
    TEST(approx_entrywise_equal(B, {12,18,18,24}));
});


static misc::UnitTest sparse_dyadicsum("SparseTensor", "sum_dyadic", [](){
    Tensor res({2,2}, Tensor::Representation::Sparse);
    Tensor B({2}, Tensor::Representation::Sparse);
    Tensor C({2}, Tensor::Representation::Sparse);

    Index i, j, k;
    
    FAILTEST(res(i,j) = B(i) + C(j));
});

static misc::UnitTest sparse_sum_three("SparseTensor", "sum_threefold", [](){
    Tensor res({2}, Tensor::Representation::Sparse);
    Tensor B({2}, Tensor::Representation::Sparse);
    Tensor C({2}, Tensor::Representation::Sparse);
    Tensor D({2}, Tensor::Representation::Sparse);

    Index i, j, k;
    
    B[0]=1;
    B[1]=2;
    
    C[0]=5;
    C[1]=9;
    
    D[0]=7;
    D[1]=13;
    
	B.use_sparse_representation();
	C.use_sparse_representation();
	D.use_sparse_representation();
	
    res(i) = B(i) + C(i) + D(i);
    TEST(approx_entrywise_equal(res, {13,24}));
});

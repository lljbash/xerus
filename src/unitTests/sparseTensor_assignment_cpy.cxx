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

static misc::UnitTest sparse_assign_triv("SparseTensor", "Assignment_Trivia", [](){
    Tensor A({2,2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res({2,2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res2({2,3,2,1,2}, Tensor::Representation::Sparse);
    Tensor res3({2,3,1,2,2}, Tensor::Representation::Sparse);

    Index i,j,k,l,m;
    
    A[{0,0,0,0,0}]=1;
    A[{0,0,0,0,1}]=2;
    A[{0,0,1,0,0}]=3;
    A[{0,0,1,0,1}]=4;
    A[{0,0,2,0,0}]=5;
    A[{0,0,2,0,1}]=6;
    A[{0,1,0,0,0}]=7;
    A[{0,1,0,0,1}]=8;
    A[{0,1,1,0,0}]=9;
    A[{0,1,1,0,1}]=10;
    A[{0,1,2,0,0}]=11;
    A[{0,1,2,0,1}]=12;
    A[{1,0,0,0,0}]=13;
    A[{1,0,0,0,1}]=14;
    A[{1,0,1,0,0}]=15;
    A[{1,0,1,0,1}]=16;
    A[{1,0,2,0,0}]=17;
    A[{1,0,2,0,1}]=18;
    A[{1,1,0,0,0}]=19;
    A[{1,1,0,0,1}]=20;
    A[{1,1,1,0,0}]=21;
    A[{1,1,1,0,1}]=22;
    A[{1,1,2,0,0}]=23;
    A[{1,1,2,0,1}]=24;
    
	A.use_sparse_representation();
	
    res(i,j,k,l,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res(i&0) = A(i&0);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res(i&0) = A(i^5);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res(i^5) = A(i&0);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res(i^3, j&3) = A(i&2,j^2);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    
    res(j,i,k,l,m) = A(j,i,k,l,m);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res(i,j,k,l,m) = A(j,i,k,l,m);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,13,14,15,16,17,18,7,8,9,10,11,12,19,20,21,22,23,24}));
    res(j,i,k,l,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,13,14,15,16,17,18,7,8,9,10,11,12,19,20,21,22,23,24}));
    
    res2(i,k,j,l,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res2, {1,2,7,8,3,4,9,10,5,6,11,12,13,14,19,20,15,16,21,22,17,18,23,24}));
    res2(i,j,k,l,m) = A(i,k,j,l,m);
    TEST(approx_entrywise_equal(res2, {1,2,7,8,3,4,9,10,5,6,11,12,13,14,19,20,15,16,21,22,17,18,23,24}));
    
    res(i,j,k,l,m) = A(i,j,l,k,m);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res(i,j,l,k,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
});


static misc::UnitTest sparse_assign_const("SparseTensor", "Assignment_Const", [](){
    Tensor A({2,2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res({2,2,3,1,2}, Tensor::Representation::Sparse);

    Index i,j,k,l,m;
    
    A[{0,0,0,0,0}]=1;
    A[{0,0,0,0,1}]=2;
    A[{0,0,1,0,0}]=3;
    A[{0,0,1,0,1}]=4;
    A[{0,0,2,0,0}]=5;
    A[{0,0,2,0,1}]=6;
    A[{0,1,0,0,0}]=7;
    A[{0,1,0,0,1}]=8;
    A[{0,1,1,0,0}]=9;
    A[{0,1,1,0,1}]=10;
    A[{0,1,2,0,0}]=11;
    A[{0,1,2,0,1}]=12;
    A[{1,0,0,0,0}]=13;
    A[{1,0,0,0,1}]=14;
    A[{1,0,1,0,0}]=15;
    A[{1,0,1,0,1}]=16;
    A[{1,0,2,0,0}]=17;
    A[{1,0,2,0,1}]=18;
    A[{1,1,0,0,0}]=19;
    A[{1,1,0,0,1}]=20;
    A[{1,1,1,0,0}]=21;
    A[{1,1,1,0,1}]=22;
    A[{1,1,2,0,0}]=23;
    A[{1,1,2,0,1}]=24;
	A.use_sparse_representation();
    
    res(i,j,k,l,m) = A(i,j,k,l,m);
    const Tensor resC1(res.sparse_copy());
    TEST(approx_entrywise_equal(resC1, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    TEST(misc::approx_equal(resC1[{0,0,0,0,0}], 1.0));
    TEST(misc::approx_equal(resC1[{0,1,2,0,0}], 11.0));
    TEST(misc::approx_equal(resC1[{1,1,2,0,0}], 23.0));
    TEST(misc::approx_equal(resC1[{1,0,0,0,1}], 14.0));
    TEST(misc::approx_equal(resC1[{1,0,1,0,1}], 16.0));
    
    res(j,i,k,l,m) = A(i,j,k,l,m);
    const Tensor resC2(res.sparse_copy());
    TEST(approx_entrywise_equal(resC2, {1,2,3,4,5,6,13,14,15,16,17,18,7,8,9,10,11,12,19,20,21,22,23,24}));
    TEST(misc::approx_equal(resC2[{0,0,0,0,0}], 1.0));
    TEST(misc::approx_equal(resC2[{0,1,2,0,0}], 17.0));
    TEST(misc::approx_equal(resC2[{1,1,2,0,0}], 23.0));
    TEST(misc::approx_equal(resC2[{1,0,0,0,1}], 8.0));
    TEST(misc::approx_equal(resC2[{1,0,1,0,1}], 10.0));
});

static misc::UnitTest sparse_assign_overwriting_dim("SparseTensor", "Assignment_Overwriting_Dimensions", [](){
    Tensor A({2,2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res1({2,2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res2({1,3,5,1,7}, Tensor::Representation::Sparse);
    Tensor res3({2,3,2,2,3}, Tensor::Representation::Sparse);
    Tensor res4({13,9,2,5,3}, Tensor::Representation::Sparse);
    Tensor res5({1,1,1,1,1}, Tensor::Representation::Sparse);
    Tensor res6( Tensor::Representation::Sparse );
    Tensor res7({1,2,3,4,5}, Tensor::Representation::Sparse);
    Tensor res8({5,4,3,2,1}, Tensor::Representation::Sparse);
    Tensor res9({3,2,3,2,3}, Tensor::Representation::Sparse);
    Tensor res10({4,5,3,1,3}, Tensor::Representation::Sparse);
    Tensor res11( Tensor::Representation::Sparse );
    Tensor res12({1,5,3,1,3}, Tensor::Representation::Sparse);

    Index i,j,k,l,m;
    
    A[{0,0,0,0,0}]=1;
    A[{0,0,0,0,1}]=2;
    A[{0,0,1,0,0}]=3;
    A[{0,0,1,0,1}]=4;
    A[{0,0,2,0,0}]=5;
    A[{0,0,2,0,1}]=6;
    A[{0,1,0,0,0}]=7;
    A[{0,1,0,0,1}]=8;
    A[{0,1,1,0,0}]=9;
    A[{0,1,1,0,1}]=10;
    A[{0,1,2,0,0}]=11;
    A[{0,1,2,0,1}]=12;
    A[{1,0,0,0,0}]=13;
    A[{1,0,0,0,1}]=14;
    A[{1,0,1,0,0}]=15;
    A[{1,0,1,0,1}]=16;
    A[{1,0,2,0,0}]=17;
    A[{1,0,2,0,1}]=18;
    A[{1,1,0,0,0}]=19;
    A[{1,1,0,0,1}]=20;
    A[{1,1,1,0,0}]=21;
    A[{1,1,1,0,1}]=22;
    A[{1,1,2,0,0}]=23;
    A[{1,1,2,0,1}]=24;
	A.use_sparse_representation();
    
    res1(i,j,k,l,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res1, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res2(i&0) = A(i&0);
    TEST(approx_entrywise_equal(res2, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res3(i&0) = A(i^5);
    TEST(approx_entrywise_equal(res3, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res4(i^5) = A(i&0);
    TEST(approx_entrywise_equal(res4, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res5(i^3, j&3) = A(i&2,j^2);
    TEST(approx_entrywise_equal(res5, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    
    res6(j,i,k,l,m) = A(j,i,k,l,m);
    TEST(approx_entrywise_equal(res6, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res7(i,j,k,l,m) = A(j,i,k,l,m);
    TEST(approx_entrywise_equal(res7, {1,2,3,4,5,6,13,14,15,16,17,18,7,8,9,10,11,12,19,20,21,22,23,24}));
    res8(j,i,k,l,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res8, {1,2,3,4,5,6,13,14,15,16,17,18,7,8,9,10,11,12,19,20,21,22,23,24}));
    
    res9(i,k,j,l,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res9, {1,2,7,8,3,4,9,10,5,6,11,12,13,14,19,20,15,16,21,22,17,18,23,24}));
    res10(i,j,k,l,m) = A(i,k,j,l,m);
    TEST(approx_entrywise_equal(res10, {1,2,7,8,3,4,9,10,5,6,11,12,13,14,19,20,15,16,21,22,17,18,23,24}));
    
    res11(i,j,k,l,m) = A(i,j,l,k,m);
    TEST(approx_entrywise_equal(res11, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
    res12(i,j,l,k,m) = A(i,j,k,l,m);
    TEST(approx_entrywise_equal(res12, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}));
});

static misc::UnitTest sparse_assign_eq("SparseTensor", "Assignment_LHS_Equals_RHS", [](){ 
    Tensor B({2,2}, Tensor::Representation::Sparse);
    Tensor C({2,2}, Tensor::Representation::Sparse);

    Index i, J, K;
    
    B[{0,0}]=1;
    B[{0,1}]=2;
    B[{1,0}]=3;
    B[{1,1}]=4;
    
    C[{0,0}]=5;
    C[{0,1}]=6;
    C[{1,0}]=7;
    C[{1,1}]=8;
    
    B(i,J) = B(i,J);
    TEST(approx_entrywise_equal(B, {1,2,3,4}));
    B(i,J) = B(J,i);
    TEST(approx_entrywise_equal(B, {1,3,2,4}));
});

static misc::UnitTest sparse_assign_fixed("SparseTensor", "Assignment_Fixed_Indices", [](){
    Tensor A({2,2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res1({2,3,1,2}, Tensor::Representation::Sparse);
    Tensor res2( Tensor::Representation::Sparse );
    Tensor res3( Tensor::Representation::Sparse );
    
    Index i,j,k,l,m;
    
    A[{0,0,0,0,0}]=1;
    A[{0,0,0,0,1}]=2;
    A[{0,0,1,0,0}]=3;
    A[{0,0,1,0,1}]=4;
    A[{0,0,2,0,0}]=5;
    A[{0,0,2,0,1}]=6;
    A[{0,1,0,0,0}]=7;
    A[{0,1,0,0,1}]=8;
    A[{0,1,1,0,0}]=9;
    A[{0,1,1,0,1}]=10;
    A[{0,1,2,0,0}]=11;
    A[{0,1,2,0,1}]=12;
    A[{1,0,0,0,0}]=13;
    A[{1,0,0,0,1}]=14;
    A[{1,0,1,0,0}]=15;
    A[{1,0,1,0,1}]=16;
    A[{1,0,2,0,0}]=17;
    A[{1,0,2,0,1}]=18;
    A[{1,1,0,0,0}]=19;
    A[{1,1,0,0,1}]=20;
    A[{1,1,1,0,0}]=21;
    A[{1,1,1,0,1}]=22;
    A[{1,1,2,0,0}]=23;
    A[{1,1,2,0,1}]=24;
	A.use_sparse_representation();
    
    res1(i,j,k,l) = A(0,i,j,k,l);
    TEST(approx_entrywise_equal(res1, {1,2,3,4,5,6,7,8,9,10,11,12}));
    res1(i,j,k,l) = A(1,i,j,k,l);
    TEST(approx_entrywise_equal(res1, {13,14,15,16,17,18,19,20,21,22,23,24}));
    
    res1(i,j,k,l) = A(i,0,j,k,l);
    TEST(approx_entrywise_equal(res1, {1,2,3,4,5,6,13,14,15,16,17,18}));
    res1(i,j,k,l) = A(i,1,j,k,l);
    TEST(approx_entrywise_equal(res1, {7,8,9,10,11,12,19,20,21,22,23,24}));
    
    res1(i,j,k,l) = A(i,j,k,l,0);
    TEST(approx_entrywise_equal(res1, {1,3,5,7,9,11,13,15,17,19,21,23}));
    res1(i,j,k,l) = A(i,j,k,l,1);
    TEST(approx_entrywise_equal(res1, {2,4,6,8,10,12,14,16,18,20,22,24}));
    
    res2(i,j,k) = A(i,j,1,k,1);
    TEST(approx_entrywise_equal(res2, {4,10,16,22}));
    res2(i,j,k) = A(i,j,2,k,1);
    TEST(approx_entrywise_equal(res2, {6,12,18,24}));
    res2(i,j,k) = A(j,i,2,k,1);
    TEST(approx_entrywise_equal(res2, {6,18,12,24}));
    
    res2(i,k,j) = A(j,i,2,k,1);
    TEST(approx_entrywise_equal(res2, {6,18,12,24}));
    
    res3(i,j) = A(j,i,2,0,1);
    TEST(approx_entrywise_equal(res3, {6,18,12,24}));
    res3(i,j) = A(1,i,2,j,1);
    TEST(approx_entrywise_equal(res3, {18,24}));
});

static misc::UnitTest sparse_assign_neg("SparseTensor", "Assignment_Negatives", [](){
    Tensor A({2,2,2,2}, Tensor::Representation::Sparse);
    Tensor A2({2,2,2,2}, Tensor::Representation::Sparse);
    Tensor B({2,2,2}, Tensor::Representation::Sparse);
    Tensor C( Tensor::Representation::Sparse );
    Tensor D({2,2}, Tensor::Representation::Sparse);
    Tensor E({}, Tensor::Representation::Sparse);
    
    Index i,j,k,l;
        
    FAILTEST(A(i) = A(i));
    FAILTEST(A(i&1) = A(i&1));
    FAILTEST(A(i^3) = A(i^3));
    FAILTEST(A(i,j^3) = A(i^3,j));
    FAILTEST(A(i^5) = A(i&0));
    FAILTEST(A(i^4,j&5) = A(i^4,j&5));
    FAILTEST(A2(i) = A(i));
    FAILTEST(A2(i&1) = A(i&1));
    FAILTEST(A2(i^3) = A(i^3));
    FAILTEST(A2(i,j^3) = A(i^3,j));
    FAILTEST(A2(i^5) = A(i&0));
    FAILTEST(A2(i^4,j&5) = A(i^4,j&5));
    FAILTEST(A(i,j,k^2) = B(i,k^2));
    FAILTEST(A(i,j,k,l) = B(i,j,k));
    FAILTEST(A(i,j,k,l) = B(j,k,i));
    FAILTEST(C(i,j) = D(i^2));
    FAILTEST(C(i,j) = D(i^2));
	static_assert(!std::is_assignable<decltype(D(i,j) * D(j,k)), decltype(D(i,k))>::value,"");
});


    
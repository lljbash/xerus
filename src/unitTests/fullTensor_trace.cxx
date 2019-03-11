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

static misc::UnitTest tensor_traces("Tensor", "Traces", [](){
    Tensor A({2,2});
    Tensor B({2,2,2});
    Tensor C({2,2,2,2});
    Tensor res1({});
    Tensor res2({2});
    Tensor res3({2,2});
    
    Index i, j, k;
    
    A[{0,0}] = 1;
    A[{0,1}] = 2;
    A[{1,0}] = 4;
    A[{1,1}] = 8;
	TEST(!A.is_sparse());
    
    B[{0,0,0}] = 1;
    B[{0,0,1}] = 2;
    B[{0,1,0}] = 4;
    B[{0,1,1}] = 8;
    B[{1,0,0}] = 16;
    B[{1,0,1}] = 32;
    B[{1,1,0}] = 64;
    B[{1,1,1}] = 128;
	TEST(!B.is_sparse());
    
    C[{0,0,0,0}] = 1;
    C[{0,0,0,1}] = 2;
    C[{0,0,1,0}] = 4;
    C[{0,0,1,1}] = 8;
    C[{0,1,0,0}] = 16;
    C[{0,1,0,1}] = 32;
    C[{0,1,1,0}] = 64;
    C[{0,1,1,1}] = 128;
    C[{1,0,0,0}] = 256;
    C[{1,0,0,1}] = 512;
    C[{1,0,1,0}] = 1024;
    C[{1,0,1,1}] = 2048;
    C[{1,1,0,0}] = 4096;
    C[{1,1,0,1}] = 8192;
    C[{1,1,1,0}] = 16384;
    C[{1,1,1,1}] = 32768;
	TEST(!C.is_sparse());
    
    res1() = A(i,i);
    MTEST(approx_entrywise_equal(res1, {9}), res1[0]);
    
    res2(j) = B(i,i,j);
    MTEST(approx_entrywise_equal(res2, {65, 130}), res2.to_string());
    res2(j) = B(i,j,i);
    MTEST(approx_entrywise_equal(res2, {33, 132}), res2.to_string());
    res2(j) = B(j,i,i);
    MTEST(approx_entrywise_equal(res2, {9, 144}), res2.to_string());
    
    res3(j,k) = C(i,i,j,k);
    MTEST(approx_entrywise_equal(res3, {4097, 8194, 16388, 32776}), res3.to_string());
    res3(j,k) = C(i,j,i,k);
    MTEST(approx_entrywise_equal(res3, {1025, 2050, 16400, 32800}), res3.to_string());
    res3(j,k) = C(i,j,k,i);
    MTEST(approx_entrywise_equal(res3, {513, 2052, 8208, 32832}), res3.to_string());
    res3(j,k) = C(j,i,i,k);
    MTEST(approx_entrywise_equal(res3, {65, 130, 16640, 33280}), res3.to_string());
    res3(j,k) = C(j,i,k,i);
    MTEST(approx_entrywise_equal(res3, {33, 132, 8448, 33792}), res3.to_string());
    res3(j,k) = C(j,k,i,i);
    MTEST(approx_entrywise_equal(res3, {9, 144, 2304, 36864}), res3.to_string());
    
    res3(k,j) = C(i,i,j,k);
    MTEST(approx_entrywise_equal(res3, {4097, 16388, 8194, 32776}), res3.to_string());
    res3(k,j) = C(i,j,i,k);
    MTEST(approx_entrywise_equal(res3, {1025, 16400, 2050, 32800}), res3.to_string());
    res3(k,j) = C(i,j,k,i);
    MTEST(approx_entrywise_equal(res3, {513, 8208, 2052, 32832}), res3.to_string());
    res3(k,j) = C(j,i,i,k);
    MTEST(approx_entrywise_equal(res3, {65, 16640, 130, 33280}), res3.to_string());
    res3(k,j) = C(j,i,k,i);
    MTEST(approx_entrywise_equal(res3, {33, 8448, 132, 33792}), res3.to_string());
    res3(k,j) = C(j,k,i,i);
    MTEST(approx_entrywise_equal(res3, {9, 2304, 144, 36864}), res3.to_string());
    
    res2(k) = C(0,k,i,i);
    MTEST(approx_entrywise_equal(res2, {9, 144}), res2.to_string());
    res2(k) = C(1,k,i,i);
    MTEST(approx_entrywise_equal(res2, {2304,36864}), res2.to_string());
    res2(j) = C(j,0,i,i);
    MTEST(approx_entrywise_equal(res2, {9, 2304}), res2.to_string());
    res2(j) = C(j,1,i,i);
    MTEST(approx_entrywise_equal(res2, {144, 36864}), res2.to_string());
    
    res1() = C(i,i,j,j);
    MTEST(approx_entrywise_equal(res1, {1+8+4096+32768}), res1.to_string());
    res1() = C(i,j,i,j);
    MTEST(approx_entrywise_equal(res1, {1+32+1024+32768}), res1.to_string());
    res1() = C(i,j,j,i);
    MTEST(approx_entrywise_equal(res1, {1+64+512+32768}), res1.to_string());
});
